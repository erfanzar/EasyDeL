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

"""eSurge Model Runner - High-performance inference execution engine.

This module implements the core execution logic for the eSurge inference engine,
providing efficient model execution with advanced features like paged attention,
dynamic batching, and compilation caching.

Key Components:
    ExecutionManager: Manages compiled execution functions for different batch/token configurations
    eSurgeRunner: Main runner class that orchestrates model execution

Architecture:
    The module uses a two-stage compilation strategy:
    1. Pre-compilation of functions for different token/batch size combinations
    2. Runtime selection of appropriate compiled function based on input shape

Performance Features:
    - Paged attention for efficient KV cache management
    - Vectorized operations for batch processing
    - Pre-allocated buffers to minimize memory allocation
    - Compilation caching to avoid recompilation
    - Progress logging for long compilation processes

Example:
    >>> from easydel.infra import EasyDeLBaseModule
    >>> from easydel.inference.esurge.runners import eSurgeRunner
    >>>
    >>> # Initialize model
    >>> model = EasyDeLBaseModule.from_pretrained("model-name")
    >>>
    >>> # Create runner
    >>> runner = eSurgeRunner(
    ...     model=model,
    ...     max_model_len=2048,
    ...     max_num_seqs=8,
    ...     hbm_utilization=0.9
    ... )
    >>>
    >>> # Compile for different configurations
    >>> runner.compile()
    >>>
    >>> # Execute model
    >>> output = runner.execute_model(scheduler_output)
"""

from __future__ import annotations

import os
import time
import typing
from bisect import bisect_left
from collections import deque
from concurrent.futures import Future
from functools import partial

import jax
import numpy as np
import spectrax as spx
from eformer.loggings import get_logger
from ejkernel.modules.operations import set_gdn_kernel_tile_policy
from jax import numpy as jnp
from jax.experimental import multihost_utils

from easydel.inference.esurge.config import (
    ESURGE_MIN_TOKEN_PAD,
    KernelTilePolicy,
)
from easydel.inference.speculative import DrafterProtocol, accept_or_reject, resample_rejected
from easydel.infra.sharding import replicated_named_sharding
from easydel.layers.quantization import TurboQuantConfig

from ..core.dp_sharding import dp_shard_page_bounds, pages_per_dp_shard
from ..core.interface import create_kv_cache_specs_from_config, estimate_runtime_page_budget
from ..metrics import get_metrics_collector
from ..outputs import ModelRunnerOutput
from ..scheduler import SchedulerOutput
from ..utils import model_uses_mrope
from .async_types import AsyncPreResults, AsyncWindowResult, DeviceInputTokenHandoff
from .execution_manager import ExecutionManager
from .host_sync import host_payload_broadcast_needed
from .perf import RunnerPerfSample, RunnerPerfTracker
from .pipeline_execution_manager import PipelineExecutionManager
from .pipeline_plan import build_pipeline_inference_plan, cap_metadata_pages
from .sequence_buffer import (
    SequenceBuffer,
    build_allowed_mask,
    build_sampling_arrays,
    fill_slice,
    move_row,
    pack_prompts,
    swap_rows,
)
from .slot_pool import RecurrentSlotPool
from .spec.interface import NullSpeculation, SpeculativeStrategy
from .spec.strategy import DrafterSpeculation, _snapshot_recurrent_state, _window_hidden_row
from .state_sync import SequenceStateSync
from .states import CachedRequestState
from .vlm_prefill import VlmPrefillHelper
from .window_planner import WindowPlanner

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule
    from easydel.infra.etils import MpMdSchedulers

logger = get_logger("eSurge")
MLA_RAGGED_ATTN_MECHANISM = "multi_latent_ragged_page_attention_v2"


class _BatchedVerifyRow(typing.NamedTuple):
    """Precomputed batched greedy verify result for one spec request.

    Populated by the per-step batched emission pre-pass (see
    ``_execute_model_impl``): the LM-head projection and greedy argmax for the
    whole running batch's verify rows run ONCE, and each request's slice is read
    back here so the per-request emission loop consumes host integers instead of
    issuing its own device->host argmax sync.

    Attributes:
        verify_meta: The request's :class:`_SpecVerifyMetadata`.
        offset: Start row of this request inside the batched gather/argmax.
        count: Number of verify rows for this request (``num_drafts + 1``).
        argmaxes: Host greedy argmax token ids for this request's ``count`` rows.
        gathered_hidden: The shared batched ``[pad_rows, H]`` gathered hidden
            device array; ``gathered_hidden[offset + accepted]`` is the seed
            hidden for drafting, bit-identical to the per-request gather.
        batched_logits: The shared batched ``[pad_rows, vocab]`` logits device
            array; sliced only when verify tracing is enabled.
    """

    verify_meta: typing.Any
    offset: int
    count: int
    argmaxes: list[int]
    gathered_hidden: jax.Array
    batched_logits: jax.Array


class _AsyncExecutionHandle:
    """Deferred host-materialized model output for overlap execution.

    Returned by :meth:`eSurgeRunner.execute_model` when overlap execution is
    enabled. Wraps a partially-materialized :class:`ModelRunnerOutput` plus
    one :class:`AsyncWindowResult` per runner window. Calling
    :meth:`get_output` finishes the host copies and returns the fully
    populated :class:`ModelRunnerOutput`.
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        windows: list[AsyncWindowResult],
        finalize: typing.Callable[[list[list[int]]], None] | None = None,
    ) -> None:
        """Initialize the deferred handle.

        Args:
            model_runner_output (ModelRunnerOutput): Output skeleton with all
                host-resolvable fields filled in (sampled tokens are filled
                later from ``windows``).
            windows (list[AsyncWindowResult]): Per-window sampled-token tensors
                whose host copies are already in flight.
            finalize: Optional callback invoked exactly once with the resolved
                ``sampled_token_ids`` once the host data is ready.
        """
        self._model_runner_output = model_runner_output
        self._windows = windows
        self._finalize = finalize
        self._resolved_output: ModelRunnerOutput | None = None

    def get_output(self) -> ModelRunnerOutput:
        """Block on host copies and return the finalized output.

        Returns:
            ModelRunnerOutput: Output with ``sampled_token_ids`` and
            ``token_logprobs`` populated. Subsequent calls return the cached
            result without redoing host transfer.
        """
        if self._resolved_output is not None:
            return self._resolved_output

        sampled_token_ids: list[list[int]] = []
        token_logprobs: dict[str, float] = {}

        for window in self._windows:
            tokens_cpu = np.asarray(window.sampled_token_ids)
            logprobs_cpu = np.asarray(window.token_logprobs) if window.token_logprobs is not None else None
            for row_pos, req_id, is_valid in zip(
                window.row_positions,
                window.req_ids,
                window.valid_mask,
                strict=False,
            ):
                if not is_valid:
                    sampled_token_ids.append([])
                    continue

                sampled_token_ids.append([int(tokens_cpu[row_pos])])
                if logprobs_cpu is not None and row_pos < logprobs_cpu.shape[0]:
                    try:
                        token_logprobs[req_id] = float(logprobs_cpu[row_pos])
                    except Exception:
                        pass

        if self._finalize is not None:
            self._finalize(sampled_token_ids)
            self._finalize = None

        output = self._model_runner_output
        output.sampled_token_ids = sampled_token_ids
        output.token_logprobs = token_logprobs or None
        self._resolved_output = output
        return output


class eSurgeRunner:
    """High-performance model runner for efficient batched inference.

    The eSurgeRunner orchestrates model execution with advanced features:
    - Paged attention for memory-efficient KV cache management
    - Dynamic batching with request scheduling
    - Pre-allocated buffers for zero-copy operations
    - Vectorized token processing
    - Compilation caching for different batch/sequence configurations

    The runner maintains an internal state of active requests and manages
    their lifecycle from prompt processing through token generation.

    Architecture:
        Request Flow:
        1. Scheduler provides requests to execute
        2. Runner updates internal state (add/remove requests)
        3. Prepares inputs with proper padding and batching
        4. Executes model using pre-compiled functions
        5. Processes sampled tokens and updates buffers
        6. Returns results to scheduler

    Memory Management:
        - Pre-allocated buffers for common operations
        - Paged KV cache with configurable page size
        - Efficient slot mapping for attention
        - Buffer reuse across batches

    Attributes:
        model: The EasyDeL model to run
        metadata: Paged attention metadata
        max_num_seqs: Maximum concurrent sequences
        max_model_len: Maximum sequence length
        executor_manager: Manages compiled functions
        sequence_buffer: Manages active sequences
        requests: Active request states

    Example:
        >>> runner = eSurgeRunner(
        ...     model=model,
        ...     max_model_len=2048,
        ...     max_num_seqs=8,
        ...     hbm_utilization=0.9,
        ...     page_size=128
        ... )
        >>>
        >>> # Compile for all configurations
        >>> runner.compile()
        >>>
        >>> # Execute requests from scheduler
        >>> output = runner.execute_model(scheduler_output)
        >>>
        >>> # Process results
        >>> for req_id, tokens in zip(output.req_ids, output.sampled_token_ids):
        ...     print(f"Request {req_id}: {tokens}")
    """

    def __init__(
        self,
        model: EasyDeLBaseModule,
        hbm_utilization: float = 0.5,
        page_size: int = 128,
        max_cache_tokens: int | None = None,
        cache_capacity_margin: float = 0.92,
        kernel_tile_policy: KernelTilePolicy = "auto",
        max_model_len: int = 2**13,
        max_num_batched_tokens: int | None = None,
        compile_vision_encoder: bool = True,
        vision_patch_buckets: list[int] | None = None,
        min_input_pad: int = 256,
        min_token_pad: int | None = None,
        max_num_seqs: int = 16,
        max_num_seq_buckets: list[int] | None = None,
        async_scheduling: bool = True,
        use_aot_forward: bool = True,
        verbose: bool = False,
        enable_overlap_execution: bool = True,
        enable_sampler_metrics: bool = False,
        enable_window_aware_runtime_cap: bool = False,
        mpmd_scheduler: MpMdSchedulers | None = None,
        pp_microbatch_count: int | str | None = "auto",
        pp_microbatch_size: int | str | None = "auto",
        drafter: DrafterProtocol | None = None,
    ):
        """Initialize the model runner.

        Args:
            model: EasyDeL model instance.
            hbm_utilization: Target cache-memory utilization ratio.
            page_size: KV page size used by cache metadata.
            max_model_len: Maximum sequence length.
            max_num_batched_tokens: Maximum scheduler token budget used for
                window-aware runtime-cap estimation.
            compile_vision_encoder: Whether to precompile and use a bucketed
                JIT helper for VLM vision features.
            vision_patch_buckets: Optional raw patch-count buckets used by the
                VLM vision precompile helper.
            min_input_pad: Minimum request-count bucket.
            min_token_pad: Optional minimum token bucket size.
            max_num_seqs: Maximum concurrent sequences.
            max_num_seq_buckets: Optional explicit request-count buckets.
            async_scheduling: Whether scheduler async token sampling is enabled.
            use_aot_forward: Whether to use AOT compilation.
            verbose: Enable verbose execution logs.
            enable_overlap_execution: Enable overlap execution path.
            enable_sampler_metrics: Enable sampler-side metrics.
            enable_window_aware_runtime_cap: Whether to derive the runtime
                request cap from the model's live KV-window page demand.
                When False, the runner falls back to the cache metadata's
                heuristic request cap instead.
            pp_microbatch_count: Expert PP decode wavefront policy. ``"auto"``
                keeps the built-in split, ``None`` or ``0`` disables the
                wavefront path, and a positive integer pins max microbatches.
            pp_microbatch_size: Expert PP decode wavefront policy. ``"auto"``
                keeps the built-in split, ``None`` or ``0`` disables the
                wavefront path, and a positive integer pins rows per
                microbatch. Mutually exclusive with a positive count.
            drafter: Optional speculative-decoding drafter implementing
                :class:`~easydel.inference.speculative.DrafterProtocol`
                (e.g. ``Qwen3_5MTPDrafter`` or ``Gemma4AssistantDrafter``).
                When set, the runner drives the runner-native
                draft/verify/commit path, filling ``request.spec_token_ids``
                from real drafts. ``None`` keeps the standard
                single-token-per-forward decode.
        """
        self.model = model.esurge_compatible_model
        # Compressed-window models (DeepSeek-V4) keep ALL decode state in
        # per-request slots (CompressedWindowCache rows). Requests must keep a
        # stable physical slot for their lifetime (SequenceBuffer rows can be
        # condensed/moved), so the recurrent slot pool is used even without
        # SPMD DP and the slot ids are threaded to the model each step.
        self._slot_indexed_state = getattr(self.model, "esurge_cache_family", None) == "compressed_window"
        if self._slot_indexed_state and drafter is not None:
            raise ValueError(
                "Speculative decoding is not supported for compressed-window cache models: "
                "per-slot ring/compressor state advances on every consumed token and cannot "
                "roll back rejected draft tokens."
            )
        self.drafter = drafter
        if self.drafter is not None and hasattr(self.drafter, "set_max_length"):
            self.drafter.set_max_length(int(max_model_len))
        self.num_speculative_tokens = int(
            getattr(drafter, "num_draft_tokens", getattr(drafter, "num_speculative_tokens", 1))
            if drafter is not None
            else 0
        )
        self.spec: SpeculativeStrategy = (
            DrafterSpeculation(
                runner=self,
                drafter=self.drafter,
                num_speculative_tokens=self.num_speculative_tokens,
            )
            if self.drafter is not None
            else NullSpeculation()
        )
        self._spec_suffix_time_acc = 0.0
        self._spec_replay_time_acc = 0.0
        logger.debug(f"Initializing eSurgeRunner with {max_model_len=}, {max_num_seqs=}")
        logger.debug(f"Configuration: {hbm_utilization=}, {page_size=}")
        self.pipeline_plan = build_pipeline_inference_plan(
            model=self.model,
            max_cache_tokens=max_cache_tokens,
            cache_capacity_margin=cache_capacity_margin,
            kernel_tile_policy=kernel_tile_policy,
        )
        set_gdn_kernel_tile_policy(self.pipeline_plan.kernel_tile_policy)

        backend = jax.default_backend()
        attn_mechanism = getattr(self.model.config.get_text_config(), "attn_mechanism", None)
        if backend == "gpu" and attn_mechanism in ("ragged_page_attention_v2", "ragged_page_attention_v3"):
            logger.warning(
                "GPU backend detected: `unified_attention` is preferred for eSurge inference; "
                f"got attn_mechanism={attn_mechanism!r}."
            )
        elif backend != "gpu" and attn_mechanism == "paged_flash_attention":
            logger.warning(
                "Paged flash attention is CUDA-only; falling back to non-CUDA backends may fail. "
                f"got backend={backend!r}."
            )
        elif backend == "tpu" and attn_mechanism == "unified_attention":
            logger.warning(
                "TPU backend detected: `ragged_page_attention_v3` is preferred for eSurge inference; "
                f"got attn_mechanism={attn_mechanism!r}."
            )
        elif backend == "tpu" and attn_mechanism == "ragged_page_attention_v2":
            logger.warning(
                "TPU backend detected: `ragged_page_attention_v3` is preferred for eSurge inference; "
                f"got attn_mechanism={attn_mechanism!r}."
            )

        if getattr(self.model.config.get_text_config(), "attn_mechanism", None) in (
            "unified_attention",
            "paged_flash_attention",
        ):
            self.metadata = self.model.create_unified_attention_cache_config(
                hbm_utilization=hbm_utilization,
                page_size=page_size,
                max_length=max_model_len,
                # Born-capped here too, so the global token cap is honored for
                # unified_attention even when cap_metadata_pages (below) is a no-op
                # outside pipeline parallelism.
                max_cache_tokens=max_cache_tokens,
            )
        else:
            self.metadata = self.model.create_ragged_page_cache_config(
                hbm_utilization=hbm_utilization,
                page_size=page_size,
                max_length=max_model_len,
                # Born-capped: honor the global token cap at creation time so the
                # page pool is never sized to the full HBM-derived count first.
                # cap_metadata_pages below still applies the capacity margin (and
                # is a no-op for the cap itself once this has already capped).
                max_cache_tokens=max_cache_tokens,
            )
        cap_metadata_pages(self.metadata, self.pipeline_plan)
        self.max_num_batched_tokens = (
            int(max_model_len)
            if max_num_batched_tokens is None
            # Per-step token budget is a SUM across batched sequences, so it may
            # legitimately exceed a single sequence's max_model_len (continuous
            # batching of multiple prefills). Don't clamp it down to max_model_len.
            else max(1, int(max_num_batched_tokens))
        )
        self.enable_window_aware_runtime_cap = bool(enable_window_aware_runtime_cap)
        self.max_model_len = max_model_len
        self.kv_cache_groups = self._build_kv_cache_groups()
        self.window_aware_runtime_estimate = self._apply_window_aware_runtime_cap(self.max_num_batched_tokens)
        self.max_num_seq_buckets = self._init_seq_buckets(max_num_seq_buckets, max_num_seqs, min_input_pad)
        self.max_num_seqs = max_num_seqs
        self.max_num_reqs = self.max_num_seq_buckets[-1]
        if self._uses_spmd_dp():
            self._validate_spmd_dp_support(attn_mechanism=attn_mechanism)
        self.async_scheduling = bool(async_scheduling)
        self.min_input_pad = max(min_input_pad, self.max_num_seq_buckets[0])
        self.page_size = int(self.metadata.page_size)
        self.max_pages_per_req = int(self.metadata.max_num_pages_per_req)

        if min_token_pad is None:
            # Keep the token bucket floor aligned with the public runtime
            # padding floor unless the caller explicitly requests a different
            # min_token_pad. This keeps startup/runtime bucket logs consistent:
            # min_input_pad=4 means the first token bucket is b4, not b1.
            min_token_pad_i = self.min_input_pad
        else:
            min_token_pad_i = int(min_token_pad)
        if int(self.max_model_len) >= ESURGE_MIN_TOKEN_PAD:
            min_token_pad_i = max(min_token_pad_i, ESURGE_MIN_TOKEN_PAD)
        min_token_pad_i = min(min_token_pad_i, int(self.max_model_len))
        self.num_tokens_paddings = self._get_token_paddings(
            min_token_size=min_token_pad_i,
            # Compile token buckets up to the per-step batched-token budget, not
            # just one sequence's length, so a multi-prefill step (sum of tokens
            # > max_token_size) still has a bucket and doesn't overflow the
            # scheduler's static budget.
            max_token_size=max(int(self.max_model_len), int(self.max_num_batched_tokens)),
            padding_gap=0,
        )
        if self.drafter is not None and int(self.num_speculative_tokens) > 0:
            # Speculative steady-state verify windows schedule EXACTLY
            # n_active * (1 + k) tokens. With power-of-two buckets those windows
            # pad 64->40, 128->80, 256->160 (~60% utilization), and most of the
            # verify forward's cost scales with the token bucket. Add an exact
            # spec bucket per request bucket so verify windows run tight.
            window = int(self.num_speculative_tokens) + 1
            max_bucket = max(int(self.max_model_len), int(self.max_num_batched_tokens))
            spec_buckets = {
                int(reqs) * window
                for reqs in self.max_num_seq_buckets
                if min_token_pad_i <= int(reqs) * window <= max_bucket
            }
            self.num_tokens_paddings = sorted(set(self.num_tokens_paddings) | spec_buckets)
        if self._uses_spmd_dp():
            dp_size = max(1, int(getattr(self.metadata, "data_parallel_size", 1) or 1))
            self.num_tokens_paddings = [int(bucket) for bucket in self.num_tokens_paddings if int(bucket) % dp_size == 0]
            if not self.num_tokens_paddings:
                raise ValueError(
                    f"Rank-major SPMD DP requires at least one token bucket divisible by DP size {dp_size}."
                )
        self.max_num_tokens = self.num_tokens_paddings[-1]
        self.window_planner = WindowPlanner(
            num_tokens_paddings=self.num_tokens_paddings,
            max_num_seq_buckets=self.max_num_seq_buckets,
        )
        self.vlm = VlmPrefillHelper(
            model_getter=lambda: self.model,
            metadata=self.metadata,
            compile_vision_encoder=compile_vision_encoder,
            vision_patch_buckets=vision_patch_buckets,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_tokens=self.max_num_tokens,
        )
        spec_full_hidden_max_tokens = 0
        if drafter is not None:
            spec_window_tokens = max(1, int(self.num_speculative_tokens) + 1)
            if bool(getattr(drafter, "supports_prefix_draft", False)):
                spec_window_tokens = max(spec_window_tokens, int(max_num_batched_tokens))
            spec_bucket_idx = bisect_left(self.num_tokens_paddings, spec_window_tokens)
            if spec_bucket_idx >= len(self.num_tokens_paddings):
                spec_bucket_idx = len(self.num_tokens_paddings) - 1
            spec_full_hidden_max_tokens = int(self.num_tokens_paddings[spec_bucket_idx])
        text_config = self.model.config.get_text_config()
        layer_types = getattr(text_config, "layer_types", ()) or ()
        recurrent_layer_types = {"linear_attention", "kda_linear_attention", "parallel_hybrid", "hybrid"}
        has_recurrent_layers = any(str(layer_type).lower() in recurrent_layer_types for layer_type in layer_types)
        self._has_recurrent_layers = bool(has_recurrent_layers)
        self.spec_decode_recurrent_candidates = bool(
            drafter is not None and int(self.num_speculative_tokens) > 0 and has_recurrent_layers
        )
        # Greedy recurrent verify: how to advance the live recurrent (GDN) state
        # after a verify window.
        #   fast path (default): reuse the fused verify forward's argmax + the
        #     deferred recurrent candidate-commit. Coherent output, ~0.95x on a
        #     27B GDN hybrid. NOT bit-identical to plain greedy (a chunked verify
        #     cannot match a per-token decode on a linear-attention model) and
        #     cross-process non-deterministic at the low bits.
        #   exact replay (EASURGE_SPEC_RECURRENT_REPLAY=1): advance the live state
        #     via a sequential-GDN replay of the accepted prefix. Bit-identical to
        #     plain greedy, but the extra per-window replay forward costs ~2.5x
        #     (measured 0.39x vs 0.95x on Qwen3.6-27B, tp=4, K=2) — so it is opt-in.
        # Default fast so speculative decoding serves its purpose (a speedup);
        # set the env to 1 when bit-exact greedy is required over throughput.
        self.spec_decode_recurrent_replay = bool(
            int(os.environ.get("EASURGE_SPEC_RECURRENT_REPLAY", "0") or 0)
        )

        logger.debug("Creating ExecutionManager and initializing pages cache")
        manager_cls = PipelineExecutionManager if self.pipeline_plan.is_enabled else ExecutionManager
        self.executor_manager = manager_cls(
            model=self.model,
            use_aot_forward=use_aot_forward,
            min_input_pad=self.min_input_pad,
            max_model_len=max_model_len,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            metadata=self.metadata,
            verbose=verbose,
            mpmd_scheduler=mpmd_scheduler,
            pipeline_plan=self.pipeline_plan,
            pp_microbatch_count=pp_microbatch_count,
            pp_microbatch_size=pp_microbatch_size,
            full_hidden_state_max_tokens=spec_full_hidden_max_tokens,
            speculative_recurrent_state_tokens=(
                int(self.num_speculative_tokens) + 1 if self.spec_decode_recurrent_candidates else 0
            ),
        )
        self.log_it = logger.info if verbose else logger.debug
        self._setup_variables()
        self.enable_overlap_execution = enable_overlap_execution
        self.enable_sampler_metrics = enable_sampler_metrics

        # Perf logging state (kept lightweight; no allocations in the hot path).
        self.perf = RunnerPerfTracker(history_maxlen=max(32768, int(max_model_len) * 4), alpha=0.2)

        # Async scheduling state
        self._pre_async_results: AsyncPreResults | None = None
        self._executor: typing.Any = None  # ThreadPoolExecutor, typed as Any to avoid circular import
        self._handoff_positions_cache: dict[int, jax.Array] = {}
        self._handoff_scalar_cache: dict[int, jax.Array] = {}
        logger.debug("eSurgeRunner initialization complete")
        self._log_startup_summary()

    def _validate_spmd_dp_support(self, *, attn_mechanism: str | None) -> None:
        """Fail early for features not wired into rank-major DP execution yet."""
        dp_size = max(1, int(getattr(self.metadata, "data_parallel_size", 1) or 1))
        if jax.default_backend() != "tpu":
            raise ValueError("Rank-major SPMD DP is currently TPU-only.")
        if dp_size <= 1:
            raise ValueError("Rank-major SPMD DP requires a data-parallel cache axis with size > 1.")
        if self.pipeline_plan.is_enabled:
            raise ValueError("Rank-major SPMD DP does not support MPMD/PP execution yet.")
        if self.drafter is not None:
            raise ValueError("Rank-major SPMD DP does not support speculative decoding yet.")
        if attn_mechanism not in ("ragged_page_attention_v3", MLA_RAGGED_ATTN_MECHANISM):
            raise ValueError(
                "Rank-major SPMD DP currently supports "
                f"ragged TPU attention only, got attn_mechanism={attn_mechanism!r}."
            )
        if int(self.max_num_reqs) % dp_size != 0:
            raise ValueError(
                "Rank-major SPMD DP requires max_num_reqs divisible by DP size: "
                f"max_num_reqs={self.max_num_reqs}, dp_size={dp_size}."
            )
        runtime_rows = min(int(self.metadata.get_max_num_seqs()), int(self.max_num_reqs))
        if runtime_rows != int(self.max_num_reqs):
            raise ValueError(
                "Rank-major SPMD DP currently requires the cache runtime request cap "
                f"to cover all request rows: runtime_rows={runtime_rows}, max_num_reqs={self.max_num_reqs}."
            )

    def _uses_spmd_dp(self) -> bool:
        """Return whether the runner is using rank-major data-parallel execution."""
        return int(getattr(self.metadata, "data_parallel_size", 1) or 1) > 1

    def _uses_recurrent_slot_pool(self) -> bool:
        """Whether requests are pinned to pooled physical state slots.

        Delegates to :meth:`RecurrentSlotPool.is_enabled`.
        """
        return self.slot_pool.is_enabled()

    def _plain_recurrent_row_sync_enabled(self) -> bool:
        """Whether device recurrent rows must track SequenceBuffer row moves.

        True only on the plain-recurrent path: the model has recurrent/SSM
        state AND requests are not pinned to a stable pooled slot. Rank-major
        SPMD DP and slot-indexed compressed-window families
        (``slot_pool.is_enabled()``) already thread ``recurrent_state_indices``
        to the kernel, so their device state is keyed by a stable slot and must
        not be permuted here.
        """
        return bool(self._has_recurrent_layers) and not self.slot_pool.is_enabled()

    def _sync_recurrent_rows(self, prev_row_owner: list[str | None] | None) -> None:
        """Move device recurrent-state rows to follow SequenceBuffer row moves.

        On the plain-recurrent path the conv/GDR/SSM cache is indexed by the
        physical ``SequenceBuffer`` row a request occupies in the packed batch.
        ``_update_states`` (condense) and ``reorder_decode_first`` can relocate a
        surviving request to a different row; without moving its device state in
        lockstep the request would read a freed/zeroed neighbour's state and emit
        garbage. This computes the row permutation from the pre-mutation layout
        ``prev_row_owner`` (which the device recurrent rows currently reflect) to
        the current layout and applies it via
        :meth:`ExecutionManager.permute_recurrent_slots`.

        Args:
            prev_row_owner: Per-row request ids captured immediately before
                ``_update_states`` this step, or ``None`` when the plain-recurrent
                row sync is disabled (non-recurrent or slot-pooled models).
        """
        if prev_row_owner is None:
            return
        n = int(self.max_num_reqs)
        new_ids = self.sequence_buffer.req_ids
        prev_index: dict[str, int] = {}
        for f, rid in enumerate(prev_row_owner):
            if rid is not None and f < n:
                prev_index[str(rid)] = f
        perm = np.full((n,), -1, dtype=np.int32)
        moved = False
        for t in range(n):
            rid = new_ids[t] if t < len(new_ids) else None
            if rid is None:
                continue
            f = prev_index.get(str(rid))
            if f is None:
                # Newly admitted this step: its row is (re)populated by prefill;
                # leaving perm[t] == -1 zeroes any stale occupant first.
                continue
            perm[t] = f
            if f != t:
                moved = True
        if not moved:
            return
        self.executor_manager.permute_recurrent_slots(perm)

    def _recurrent_rows_per_dp_rank(self) -> int:
        """Return the number of recurrent-state rows owned by each DP rank.

        Delegates to :meth:`RecurrentSlotPool.rows_per_dp_rank`.
        """
        return self.slot_pool.rows_per_dp_rank()

    def _reset_recurrent_slot_pools(self) -> None:
        """Initialize physical recurrent-state slots partitioned by DP rank.

        Delegates to :meth:`RecurrentSlotPool.reset`.
        """
        self.slot_pool.reset()

    def _assign_recurrent_slot(self, req_id: str, dp_rank: int | None) -> int | None:
        """Assign or return the stable physical recurrent-state slot for a request.

        Delegates to :meth:`RecurrentSlotPool.assign_slot`.
        """
        return self.slot_pool.assign_slot(req_id, dp_rank)

    def _release_recurrent_slot(self, req_id: str, *, forget_rank: bool) -> int | None:
        """Release a request's physical recurrent-state slot and return it for clearing.

        Delegates to :meth:`RecurrentSlotPool.release_slot`.
        """
        return self.slot_pool.release_slot(req_id, forget_rank=forget_rank)

    @property
    def _recurrent_slot_by_req(self) -> dict[str, int]:
        """Live request-id to physical-slot map (see :class:`RecurrentSlotPool`)."""
        return self.slot_pool.slot_by_req

    @property
    def _request_dp_rank_by_req(self) -> dict[str, int]:
        """Live request-id to DP-rank map (see :class:`RecurrentSlotPool`)."""
        return self.slot_pool.dp_rank_by_req

    @property
    def _free_recurrent_slots_by_rank(self) -> list[list[int]]:
        """Per-rank free-lists of physical slots (see :class:`RecurrentSlotPool`)."""
        return self.slot_pool.free_slots_by_rank

    def _build_kv_cache_groups(self):
        """Build cache-group specs for runtime-cap and scheduler estimation.

        Inspects the model's text config to determine KV head count and head
        dimension, then delegates to ``create_kv_cache_specs_from_config`` to
        produce one ``CacheGroupSpec`` per distinct attention type. MLA models
        return an empty list because their cache layout is handled separately.

        Returns:
            List of ``CacheGroupSpec`` objects, one per attention type group.
            Empty for MLA-based models.
        """

        text_config = self.model.config.get_text_config()
        attn_mechanism = str(getattr(text_config, "attn_mechanism", "") or "").lower()
        if "multi_latent" in attn_mechanism:
            return []

        metadata = self.metadata
        num_kv_heads = getattr(text_config, "num_kv_heads", None)
        if isinstance(num_kv_heads, (list, tuple)):
            num_kv_heads = int(num_kv_heads[0]) if len(num_kv_heads) > 0 else None
        if num_kv_heads is None:
            num_kv_heads = getattr(text_config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(text_config, "num_attention_heads", None)
        if num_kv_heads is None or int(num_kv_heads) <= 0:
            num_kv_heads = getattr(metadata, "num_kv_heads", 1)

        head_size = getattr(text_config, "head_dim", None)
        if head_size is None or int(head_size) <= 0:
            hidden_size = getattr(text_config, "hidden_size", None)
            num_attention_heads = getattr(text_config, "num_attention_heads", None)
            if hidden_size and num_attention_heads:
                head_size = int(hidden_size) // int(num_attention_heads)
        if head_size is None or int(head_size) <= 0:
            head_size = getattr(metadata, "k_headdim", None) or getattr(metadata, "head_dim", None) or 1

        return create_kv_cache_specs_from_config(
            config=text_config,
            page_size=int(metadata.page_size),
            num_kv_heads=int(num_kv_heads),
            head_size=int(head_size),
            dtype=metadata.kvdtype,
            use_mla=False,
        )

    def _get_full_attention_page_table_index(self) -> int:
        """Return the page table group index for the full-attention cache group.

        For mixed-attention models (e.g., sliding window + full attention), the
        kernel must receive the full-attention group's page table because it
        keeps all pages valid. Sliding-window groups evict old pages, leaving
        null entries that would cause VMEM out-of-range errors on TPU.

        Returns:
            0 if no cache groups are defined (single-group model), otherwise
            the index of the first FullAttentionSpec group.
        """
        from ..core.interface import FullAttentionSpec

        for i, group in enumerate(self.kv_cache_groups):
            if isinstance(group.kv_cache_spec, FullAttentionSpec):
                return i
        return 0

    def _clear_window_aware_runtime_cap_metadata(self) -> None:
        """Reset the cache-metadata fields that the window-aware estimate writes.

        Sentinel-clears the three derived attributes
        (``window_aware_max_num_seqs``, ``window_aware_pages_per_request``,
        ``window_aware_max_num_batched_tokens``) on ``self.metadata`` to
        ``-1`` so downstream consumers (the scheduler's heuristic
        request-cap path) treat them as absent. Always called before
        writing fresh values in :meth:`_apply_window_aware_runtime_cap`,
        and used as the no-op path when window-aware estimation is
        disabled.
        """
        for attr_name in (
            "window_aware_max_num_seqs",
            "window_aware_pages_per_request",
            "window_aware_max_num_batched_tokens",
        ):
            if hasattr(self.metadata, attr_name):
                setattr(self.metadata, attr_name, -1)

    def _apply_window_aware_runtime_cap(self, max_num_batched_tokens: int):
        """Attach a hybrid full/sliding runtime-cap estimate to cache metadata.

        Calls ``estimate_runtime_page_budget`` using the runner's page pool and
        cache groups, then writes the resulting concurrency limits back onto
        ``self.metadata`` so the scheduler can use them.

        Args:
            max_num_batched_tokens: Maximum number of tokens batched in one
                decode step; used to size each request's page demand.

        Returns:
            The ``RuntimePageBudgetEstimate`` if estimation succeeds, or
            ``None`` if no cache groups are available or an error occurs.
        """
        self._clear_window_aware_runtime_cap_metadata()

        if not self.enable_window_aware_runtime_cap:
            logger.debug("Window-aware runtime-cap estimation disabled; using heuristic request caps.")
            return None

        if not self.kv_cache_groups:
            return None

        try:
            estimate = estimate_runtime_page_budget(
                num_pages=int(getattr(self.metadata, "num_pages", 0) or 0),
                kv_cache_groups=list(self.kv_cache_groups),
                max_model_len=int(self.max_model_len),
                max_num_batched_tokens=int(max_num_batched_tokens),
                data_parallel_size=int(getattr(self.metadata, "data_parallel_size", 1) or 1),
            )
        except Exception as exc:
            logger.debug("Window-aware runtime-cap estimation skipped: %s", exc, exc_info=True)
            return None

        self.metadata.window_aware_max_num_seqs = int(estimate.max_num_seqs)
        self.metadata.window_aware_pages_per_request = int(estimate.pages_per_request)
        self.metadata.window_aware_max_num_batched_tokens = int(max_num_batched_tokens)
        return estimate

    def _log_startup_summary(self) -> None:
        """Log a consolidated startup summary to the logger.

        Inspects the model configuration to gather architecture details
        (layer types, attention mechanism), cache configuration (page count,
        sequence capacity), and recurrent operation names, then emits a
        single multi-line INFO log with all key runtime parameters.
        """
        try:
            text_config = self.model.config.get_text_config()
            model_type = getattr(text_config, "model_type", "unknown")
            attn_mechanism = getattr(text_config, "attn_mechanism", "unknown")
            num_layers = getattr(text_config, "num_hidden_layers", 0)
            layer_types = getattr(text_config, "layer_types", None)
            cache_info = None
            try:
                cache_info = self.model.get_operations_cache_info()
            except Exception:
                pass

            rec_ops: set[str] = set()
            if cache_info is not None and len(cache_info.layers) > 0:
                for layer in cache_info.layers:
                    if layer.is_recurrent_layer:
                        rec_ops.add(layer.operation_name)
                cache_type = cache_info.get_recommended_cache_type()
            else:
                cache_type = "paged"

            if layer_types is not None:
                from collections import Counter

                type_counts = Counter(layer_types)
                n_attn = sum(v for k, v in type_counts.items() if "full" in k or "sliding" in k)
                n_linear = sum(v for k, v in type_counts.items() if "linear" in k)
                n_parallel = type_counts.get("parallel_hybrid", 0)
                n_other = num_layers - n_attn - n_linear - n_parallel

                parts = []
                if n_parallel:
                    parts.append(f"{n_parallel} parallel attn+ssm")
                if n_linear:
                    parts.append(f"{n_linear} linear")
                if n_attn:
                    parts.append(f"{n_attn} full-attention")
                if n_other:
                    parts.append(f"{n_other} other")

                has_recurrent = n_linear > 0 or n_parallel > 0
                has_attention = n_attn > 0 or n_parallel > 0

                if n_parallel and not n_attn and not n_linear:
                    arch_desc = f"parallel_hybrid ({' + '.join(parts)} / {num_layers} layers)"
                elif has_recurrent and has_attention:
                    arch_desc = f"hybrid ({' + '.join(parts)} / {num_layers} layers)"
                elif has_recurrent and not has_attention:
                    arch_desc = f"recurrent ({' + '.join(parts)} / {num_layers} layers)"
                else:
                    arch_desc = f"attention ({num_layers} layers)"
            elif num_layers > 0:
                arch_desc = f"attention ({num_layers} layers)"
            else:
                arch_desc = "unknown"

            algos = [f"attention={attn_mechanism}"]
            if rec_ops:
                algos.append(f"linear={', '.join(sorted(rec_ops))}")
            algo_str = " | ".join(algos)

            cache_parts = [f"type={cache_type}"]
            if hasattr(self.metadata, "num_pages") and hasattr(self.metadata, "page_size"):
                n_pages = int(self.metadata.num_pages)
                p_size = int(self.metadata.page_size)
                seq_cap = int((n_pages * p_size) / 1000)
                cache_parts.append(f"pages={n_pages:,} ({p_size} tok/page)")
                cache_parts.append(f"sequence_capacity={seq_cap:,}K")
            if self.pipeline_plan.is_enabled:
                cache_parts.append(f"pp_stages={self.pipeline_plan.mpmd_dim}")
                cache_parts.append(f"pp_cache_layers/stage={self.pipeline_plan.max_stage_cache_layers}")
            window_pages_per_req = int(getattr(self.metadata, "window_aware_pages_per_request", -1) or -1)
            if window_pages_per_req > 0:
                cache_parts.append(f"pages/request={window_pages_per_req}")
            if hasattr(self.metadata, "get_max_num_seqs"):
                try:
                    max_len_cap = min(int(self.metadata.get_max_num_seqs()), int(self.max_num_reqs))
                    cache_parts.append(f"max_len_concurrency={max_len_cap:,} reqs")
                except Exception:
                    logger.debug("Could not compute runtime concurrency summary", exc_info=True)

            lines = [
                f"Model : {model_type}",
                f"Architecture : {arch_desc}",
                f"Algorithms : {algo_str}",
                f"Cache : {' | '.join(cache_parts)}",
            ]
            logger.info("\n".join(lines))
        except Exception as e:
            logger.debug(f"Could not generate startup summary: {e}")

    # Perf-tracker passthroughs
    # The perf state lives on ``self.perf`` (:class:`RunnerPerfTracker`). The
    # historical ``_perf_*`` attribute names are kept as thin properties because
    # both the execute path and external readers (scripts/bench_esurge.py,
    # serving benchmarks) reference them directly.

    @property
    def _perf_iteration(self) -> int:
        """Monotonically increasing step counter (see :class:`RunnerPerfTracker`)."""
        return self.perf.iteration

    @_perf_iteration.setter
    def _perf_iteration(self, value: int) -> None:
        self.perf.iteration = value

    @property
    def _perf_tps_ema(self) -> float | None:
        """EMA of aggregate tokens/second (see :class:`RunnerPerfTracker`)."""
        return self.perf.tps_ema

    @_perf_tps_ema.setter
    def _perf_tps_ema(self, value: float | None) -> None:
        self.perf.tps_ema = value

    @property
    def _perf_alpha(self) -> float:
        """EMA smoothing factor (see :class:`RunnerPerfTracker`)."""
        return self.perf.alpha

    @_perf_alpha.setter
    def _perf_alpha(self, value: float) -> None:
        self.perf.alpha = value

    @property
    def _perf_last_agg_tps(self) -> float | None:
        """Aggregate tokens/second of the most recent step."""
        return self.perf.last_agg_tps

    @_perf_last_agg_tps.setter
    def _perf_last_agg_tps(self, value: float | None) -> None:
        self.perf.last_agg_tps = value

    @property
    def _perf_last_req_tps(self) -> float | None:
        """Per-request tokens/second of the most recent step."""
        return self.perf.last_req_tps

    @_perf_last_req_tps.setter
    def _perf_last_req_tps(self, value: float | None) -> None:
        self.perf.last_req_tps = value

    @property
    def _perf_last_total_time(self) -> float | None:
        """Wall-clock seconds of the most recent step."""
        return self.perf.last_total_time

    @_perf_last_total_time.setter
    def _perf_last_total_time(self, value: float | None) -> None:
        self.perf.last_total_time = value

    @property
    def _perf_last_total_tokens(self) -> int | None:
        """Tokens fed to the model on the most recent step."""
        return self.perf.last_total_tokens

    @_perf_last_total_tokens.setter
    def _perf_last_total_tokens(self, value: int | None) -> None:
        self.perf.last_total_tokens = value

    @property
    def _perf_history(self) -> deque[RunnerPerfSample]:
        """Bounded deque of :class:`RunnerPerfSample` step snapshots."""
        return self.perf.history

    @property
    def _perf_phase_history(self) -> deque[dict[str, typing.Any]]:
        """Bounded deque of per-step phase-timing dicts (read by bench_esurge)."""
        return self.perf.phase_history

    # Speculative-decoding passthroughs
    # The draft/verify/commit machinery lives on ``self.spec`` (NullSpeculation or
    # DrafterSpeculation). The historical runner attribute names are kept as thin
    # delegating properties so the engine, benchmarks, and external tests are
    # untouched.

    @property
    def spec_decode_num_drafts_generated(self) -> int:
        """Total draft tokens proposed (delegates to ``self.spec``)."""
        return int(self.spec.num_drafts_generated)

    @spec_decode_num_drafts_generated.setter
    def spec_decode_num_drafts_generated(self, value: int) -> None:
        self.spec.num_drafts_generated = int(value)

    @property
    def spec_decode_num_drafts_accepted(self) -> int:
        """Total draft tokens accepted by verification (delegates to ``self.spec``)."""
        return int(self.spec.num_drafts_accepted)

    @spec_decode_num_drafts_accepted.setter
    def spec_decode_num_drafts_accepted(self, value: int) -> None:
        self.spec.num_drafts_accepted = int(value)

    @property
    def spec_decode_num_verify_steps(self) -> int:
        """Total speculative verify windows processed (delegates to ``self.spec``)."""
        return int(self.spec.num_verify_steps)

    @spec_decode_num_verify_steps.setter
    def spec_decode_num_verify_steps(self, value: int) -> None:
        self.spec.num_verify_steps = int(value)

    @property
    def spec_decode_reject_backoff_steps(self) -> int:
        """Drafter-call backoff after a full rejection (delegates to ``self.spec``)."""
        return int(self.spec.reject_backoff_steps)

    @spec_decode_reject_backoff_steps.setter
    def spec_decode_reject_backoff_steps(self, value: int) -> None:
        self.spec.reject_backoff_steps = int(value)

    @property
    def spec_decode_debug_traces(self) -> list[dict[str, typing.Any]]:
        """Collected speculative verify traces (delegates to ``self.spec``)."""
        return self.spec.debug_traces

    @property
    def spec_decode_debug_max_traces(self) -> int:
        """Speculative verify-trace budget (delegates to ``self.spec``)."""
        return int(self.spec.debug_max_traces)

    @spec_decode_debug_max_traces.setter
    def spec_decode_debug_max_traces(self, value: int) -> None:
        self.spec.debug_max_traces = int(value)

    @property
    def mesh(self):
        """The model's JAX/Spectrax sharding mesh.

        Surfaced as a property so the runner code can keep referring to
        ``self.mesh`` even as the underlying model reference is swapped
        during a hot weight update.

        Returns:
            The :class:`MeshLike` mesh the model was built on; may be a
            standard JAX mesh or an MPMD ``MpMdMesh`` when pipeline
            parallelism is active.
        """
        return self.model.mesh

    @property
    def _empty_sharding(self):
        """Cheap fully-replicated ``NamedSharding`` for scalar-shaped placement.

        Used by the runner whenever a tensor needs to live on the mesh
        but has no axis to shard along (perf scalars, host-prepared scratch
        buffers, etc.). Building the sharding fresh per call keeps it cheap
        and side-effect-free.

        Returns:
            ``NamedSharding(self.mesh, PartitionSpec())`` — replicated on
            every device in the mesh.
        """
        return replicated_named_sharding(self.mesh)

    @staticmethod
    def _get_token_paddings(min_token_size: int, max_token_size: int, padding_gap: int) -> list[int]:
        """Generate padding sizes for efficient compilation.

        Delegates to :meth:`WindowPlanner.get_token_paddings`.
        """
        return WindowPlanner.get_token_paddings(min_token_size, max_token_size, padding_gap)

    @staticmethod
    def _get_request_paddings(min_bucket: int, max_bucket: int) -> list[int]:
        """Generate request count buckets using exponential growth.

        Delegates to :meth:`WindowPlanner.get_request_paddings`.
        """
        return WindowPlanner.get_request_paddings(min_bucket, max_bucket)

    def _init_seq_buckets(
        self,
        user_buckets: list[int] | None,
        max_num_seqs: int,
        min_input_pad: int,
    ) -> list[int]:
        """Initialize sequence count buckets for compilation.

        Delegates to :meth:`WindowPlanner.init_seq_buckets`.
        """
        return WindowPlanner.init_seq_buckets(user_buckets, max_num_seqs, min_input_pad)

    def _get_current_bucket(self, num_reqs: int) -> int:
        """Select the smallest bucket that can accommodate num_reqs.

        Delegates to :meth:`WindowPlanner.get_current_bucket`, passing the
        runtime-clamped bucket list when it has been set up.
        """
        return self.window_planner.get_current_bucket(num_reqs, getattr(self, "active_num_seq_buckets", None))

    @staticmethod
    def _clamp_request_buckets_to_runtime_cap(buckets: list[int], runtime_cap: int) -> list[int]:
        """Clamp request-count buckets to the runtime execution cap.

        Delegates to :meth:`WindowPlanner.clamp_request_buckets_to_runtime_cap`.
        """
        return WindowPlanner.clamp_request_buckets_to_runtime_cap(buckets, runtime_cap)

    def _setup_variables(self):
        """Initialize internal variables and preallocate reusable buffers.

        Computes the runtime request cap from paged-attention metadata,
        clamps sequence buckets accordingly, creates the ``SequenceBuffer``
        for tracking active sequences, and allocates fixed JAX arrays
        (``input_ids_buf``, ``position_ids_buf``, ``arange``, etc.) that
        are reused across iterations to avoid repeated allocation.
        """
        self.num_reqs_max_model_len = min(self.metadata.get_max_num_seqs(), self.max_num_reqs)
        self.num_reqs_most_model_len = self.num_reqs_max_model_len
        self._allow_sparse_window_packing = (
            int(getattr(self.metadata, "data_parallel_size", 1) or 1) <= 1 and not self.async_scheduling
        )
        self.active_num_seq_buckets = self._clamp_request_buckets_to_runtime_cap(
            self.max_num_seq_buckets,
            self.num_reqs_max_model_len,
        )
        self.requests: dict[str, CachedRequestState] = {}
        logger.debug(f"Token padding sizes: {len(self.num_tokens_paddings)} levels, max={self.max_num_tokens}")
        logger.debug(
            "Active request buckets clamped to runtime cap: %s (configured=%s, runtime_cap=%s)",
            self.active_num_seq_buckets,
            self.max_num_seq_buckets,
            self.num_reqs_max_model_len,
        )
        logger.debug("Sparse zero-token row packing enabled: %s", self._allow_sparse_window_packing)

        logger.debug(
            f"Creating sequence buffer for max_num_reqs={self.max_num_reqs}, max_model_len={self.max_model_len}"
        )
        num_cache_groups = max(1, len(self.kv_cache_groups))
        self.sequence_buffer = SequenceBuffer(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            vocab_size=self.model.config.get_text_config().vocab_size,
            page_sizes=[self.metadata.page_size] * num_cache_groups,
            sharding=self._empty_sharding,
        )
        self.slot_pool = RecurrentSlotPool(
            metadata=self.metadata,
            max_num_reqs=self.max_num_reqs,
            slot_indexed_state=self._slot_indexed_state,
        )

        self.arange = jnp.arange(self.max_num_tokens, dtype=jnp.int32)
        self.arange_np = jnp.arange(self.max_num_reqs, dtype=jnp.int32)

        self.input_ids_buf = jnp.zeros((self.max_num_tokens,), dtype=jnp.int32, device=self._empty_sharding)
        self.position_ids_buf = jnp.zeros((self.max_num_tokens,), dtype=jnp.int32, device=self._empty_sharding)
        self.num_tokens_paddings_arr = jnp.array(self.num_tokens_paddings, dtype=jnp.int32, device=self._empty_sharding)
        self.scheduled_full_buf = jnp.zeros((self.max_num_reqs,), dtype=jnp.int32, device=self._empty_sharding)
        self.req_num_tokens_full_buf = jnp.zeros((self.max_num_reqs,), dtype=jnp.int32, device=self._empty_sharding)
        self.active_mask_full_buf = jnp.zeros((self.max_num_reqs,), dtype=bool, device=self._empty_sharding)

        # Host-side scratch buffers (avoid per-step NumPy allocations in hot path).
        self._scheduled_full_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._active_mask_full_cpu = np.zeros((self.max_num_reqs,), dtype=bool)
        self._req_num_tokens_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._window_temperature_cpu = np.zeros_like(self.sequence_buffer.temperature)
        self._window_top_p_cpu = np.zeros_like(self.sequence_buffer.top_p)
        self._window_top_k_cpu = np.zeros_like(self.sequence_buffer.top_k)
        self._window_min_p_cpu = np.zeros_like(self.sequence_buffer.min_p)
        self._window_frequency_penalties_cpu = np.zeros_like(self.sequence_buffer.frequency_penalties)
        self._window_presence_penalties_cpu = np.zeros_like(self.sequence_buffer.presence_penalties)
        self._window_repetition_penalties_cpu = np.ones_like(self.sequence_buffer.repetition_penalties)
        self._window_row_indices_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._window_recurrent_slot_indices_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._spec_recurrent_commit_cpu = np.zeros((2, self.max_num_reqs), dtype=np.int32)
        self._pending_spec_recurrent_commit_by_req: dict[str, int] = {}
        self.executor_manager.invalidate_sampler_penalty_state(
            self.sequence_buffer.token_ids,
            self.sequence_buffer.num_tokens,
        )

        # VLM host-side scratch buffers keyed by `num_tokens_static` (avoid repeated
        # large allocations while keeping the step-function input pytree stable).
        self.vlm.reset_cpu_buffers()
        self.state_sync = SequenceStateSync(
            sequence_buffer=self.sequence_buffer,
            requests=self.requests,
            slot_pool=self.slot_pool,
            metadata=self.metadata,
            executor_manager=self.executor_manager,
            max_model_len=self.max_model_len,
            spec_decode_recurrent_candidates=self.spec_decode_recurrent_candidates,
            spec_recurrent_commit_cpu=self._spec_recurrent_commit_cpu,
            pending_spec_recurrent_commit_by_req=self._pending_spec_recurrent_commit_by_req,
        )

    def _get_vlm_cpu_buffers(
        self,
        *,
        num_tokens_static: int,
        uses_mrope_model: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[np.ndarray] | None]:
        """Get or create cached CPU buffers for VLM prefill data.

        Delegates to :meth:`VlmPrefillHelper.get_cpu_buffers`.
        """
        return self.vlm.get_cpu_buffers(
            num_tokens_static=num_tokens_static,
            uses_mrope_model=uses_mrope_model,
        )

    def _get_window_state_views(
        self,
        *,
        start_index: int,
        row_count: int,
        page_table_cpu: np.ndarray,
        page_table_version: int | None,
        row_indices: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int | None,
    ]:
        """Return CPU-side state views aligned to the active scheduler window.

        Slices row-backed request state for the current scheduling window and
        copies per-request sampling scalars into fixed-size scratch buffers so
        downstream batch-preparation code still receives arrays sized for
        ``max_num_reqs``. For nonzero windows the page-table version is salted
        with ``start_index`` to avoid cache collisions between different row
        slices that share the same underlying page-table allocation. Packed
        non-contiguous row selections disable page-table cache reuse to avoid
        stale cache hits on mismatched row layouts.
        """
        row_count = max(0, int(row_count))
        start_index = max(0, int(start_index))
        end_index = start_index + row_count
        if row_indices is not None:
            row_indices = np.asarray(row_indices, dtype=np.int32)
            token_ids_window_cpu = self.sequence_buffer.token_ids[row_indices]
            num_computed_tokens_window_cpu = self.sequence_buffer.num_computed_tokens[row_indices]
            page_table_window_cpu = page_table_cpu[row_indices]
            start_index = int(row_indices[0]) if row_indices.size else 0
            packed_rows = True
        else:
            token_ids_window_cpu = self.sequence_buffer.token_ids[start_index:end_index]
            num_computed_tokens_window_cpu = self.sequence_buffer.num_computed_tokens[start_index:end_index]
            page_table_window_cpu = page_table_cpu[start_index:end_index]
            packed_rows = False

        temperature_window_cpu = self._window_temperature_cpu
        top_p_window_cpu = self._window_top_p_cpu
        top_k_window_cpu = self._window_top_k_cpu
        min_p_window_cpu = self._window_min_p_cpu
        frequency_penalties_window_cpu = self._window_frequency_penalties_cpu
        presence_penalties_window_cpu = self._window_presence_penalties_cpu
        repetition_penalties_window_cpu = self._window_repetition_penalties_cpu

        temperature_window_cpu.fill(0)
        top_p_window_cpu.fill(1.0)
        top_k_window_cpu.fill(0)
        min_p_window_cpu.fill(0)
        frequency_penalties_window_cpu.fill(0.0)
        presence_penalties_window_cpu.fill(0.0)
        repetition_penalties_window_cpu.fill(1.0)

        if row_count > 0:
            if row_indices is not None:
                temperature_window_cpu[:row_count] = self.sequence_buffer.temperature[row_indices]
                top_p_window_cpu[:row_count] = self.sequence_buffer.top_p[row_indices]
                top_k_window_cpu[:row_count] = self.sequence_buffer.top_k[row_indices]
                min_p_window_cpu[:row_count] = self.sequence_buffer.min_p[row_indices]
                frequency_penalties_window_cpu[:row_count] = self.sequence_buffer.frequency_penalties[row_indices]
                presence_penalties_window_cpu[:row_count] = self.sequence_buffer.presence_penalties[row_indices]
                repetition_penalties_window_cpu[:row_count] = self.sequence_buffer.repetition_penalties[row_indices]
            else:
                temperature_window_cpu[:row_count] = self.sequence_buffer.temperature[start_index:end_index]
                top_p_window_cpu[:row_count] = self.sequence_buffer.top_p[start_index:end_index]
                top_k_window_cpu[:row_count] = self.sequence_buffer.top_k[start_index:end_index]
                min_p_window_cpu[:row_count] = self.sequence_buffer.min_p[start_index:end_index]
                frequency_penalties_window_cpu[:row_count] = self.sequence_buffer.frequency_penalties[
                    start_index:end_index
                ]
                presence_penalties_window_cpu[:row_count] = self.sequence_buffer.presence_penalties[
                    start_index:end_index
                ]
                repetition_penalties_window_cpu[:row_count] = self.sequence_buffer.repetition_penalties[
                    start_index:end_index
                ]

        # The batch-preparer page-table cache key must distinguish different
        # row windows that share the same underlying page-table version.
        if page_table_version is None:
            page_table_window_version = None
        elif packed_rows:
            page_table_window_version = None
        elif start_index == 0:
            page_table_window_version = int(page_table_version)
        else:
            page_table_window_version = int(page_table_version) * (int(self.max_num_reqs) + 1) + start_index

        return (
            token_ids_window_cpu,
            num_computed_tokens_window_cpu,
            temperature_window_cpu,
            top_p_window_cpu,
            top_k_window_cpu,
            min_p_window_cpu,
            page_table_window_cpu,
            frequency_penalties_window_cpu,
            presence_penalties_window_cpu,
            repetition_penalties_window_cpu,
            page_table_window_version,
        )

    def _collect_schedulable_window_rows(
        self,
        *,
        start_index: int,
        stop_index: int,
        scheduled_tokens_by_req: dict[str, int],
        allow_sparse_packing: bool,
    ) -> tuple[np.ndarray, list[str | None], list[int], int, bool]:
        """Collect runnable rows for a window, compacting interior zero-token gaps.

        Delegates to :meth:`WindowPlanner.collect_schedulable_window_rows` with
        the live sequence-buffer row ids.
        """
        return WindowPlanner.collect_schedulable_window_rows(
            req_ids=self.sequence_buffer.req_ids,
            start_index=start_index,
            stop_index=stop_index,
            scheduled_tokens_by_req=scheduled_tokens_by_req,
            allow_sparse_packing=allow_sparse_packing,
        )

    def _precompile_jitted_helpers(
        self,
        reqs_padds: list[int],
        prompt_len_buckets: list[int],
        precompile_allowed_mask: bool = False,
        allowed_max: int = 512,
    ) -> None:
        """Precompile JIT helper kernels for various input configurations.

        Compiles auxiliary JIT functions (pack_prompts, build_sampling_arrays,
        fill_slice, swap_rows, move_row, build_allowed_mask) for different
        request and prompt length combinations to avoid runtime compilation.

        Args:
            reqs_padds: List of request count bucket sizes to compile.
            prompt_len_buckets: List of prompt length bucket sizes to compile.
            precompile_allowed_mask: Whether to compile allowed mask kernel.
            allowed_max: Maximum allowed token count for constrained decoding.

        Note:
            Compilation failures are logged at debug level and skipped,
            allowing partial precompilation when some configurations are
            not supported by the underlying kernels.
        """
        logger.info("Precompiling eSurgeRunner helper kernels")

        B = self.max_num_reqs
        T = self.max_model_len
        V = int(self.model.config.get_text_config().vocab_size)

        token_ids = jnp.zeros((B, T), dtype=jnp.int32)
        num_prompt_tokens = jnp.zeros((B,), dtype=jnp.int32)

        temperature = jnp.zeros((B,), dtype=jnp.float32)
        min_p = jnp.zeros((B,), dtype=jnp.float32)
        top_p = jnp.ones((B,), dtype=jnp.float32)
        top_k = jnp.zeros((B,), dtype=jnp.int32)

        for pr_len in prompt_len_buckets:
            pr_len = min(pr_len, self.max_model_len)
            for pr_reqs in reqs_padds:
                try:
                    lowered = pack_prompts.lower(  # pyright: ignore[reportFunctionMemberAccess]
                        token_ids,
                        num_prompt_tokens,
                        padded_num_reqs=pr_reqs,
                        padded_prompt_len=pr_len,
                        pad_id=V,
                    )
                    _ = lowered.compile()
                    logger.debug(f"pack_prompts compiled for (padded_num_reqs={pr_reqs}, padded_prompt_len={pr_len})")
                except Exception as e:
                    logger.debug(f"pack_prompts skip ({pr_reqs}, {pr_len}): {e}")

        for pr_reqs in reqs_padds:
            try:
                lowered = build_sampling_arrays.lower(  # pyright: ignore[reportFunctionMemberAccess]
                    temperature,
                    min_p,
                    top_p,
                    top_k,
                    jnp.int32(min(pr_reqs, B)),  # num_reqs <= padded_num_reqs
                    padded_num_reqs=pr_reqs,
                )
                _ = lowered.compile()
                logger.debug(f"build_sampling_arrays compiled for (padded_num_reqs={pr_reqs})")
            except Exception as e:
                logger.debug(f"build_sampling_arrays skip ({pr_reqs}): {e}")

        for pr_reqs in reqs_padds:
            try:
                lowered = fill_slice.lower(  # pyright: ignore[reportFunctionMemberAccess]
                    temperature,
                    jnp.float32(0.0),
                    int(pr_reqs),
                    int(pr_reqs),
                )
                _ = lowered.compile()
                logger.debug(f"fill_slice compiled for (num_reqs={pr_reqs}, padded_num_reqs={pr_reqs})")
            except Exception as e:
                logger.debug(f"fill_slice skip ({pr_reqs}): {e}")

        try:
            _ = swap_rows.lower(token_ids, jnp.int32(0), jnp.int32(1)).compile()
            _ = move_row.lower(token_ids, jnp.int32(0), jnp.int32(1)).compile()
            logger.debug("swap_rows and move_row compiled")
        except Exception as e:
            logger.debug(f"swap_rows/move_row skip: {e}")

        if precompile_allowed_mask:
            max_allowed = int(min(allowed_max, V))
            allowed_ids_padded = jnp.zeros((B, max_allowed), dtype=jnp.int32)
            allowed_lens = jnp.zeros((B,), dtype=jnp.int32)
            try:
                lowered = build_allowed_mask.lower(  # pyright: ignore[reportFunctionMemberAccess]
                    allowed_ids_padded,
                    allowed_lens,
                    vocab_size=int(V),
                    max_allowed=max_allowed,
                )
                _ = lowered.compile()
                logger.debug(f"build_allowed_mask compiled for (B={B}, V={V}, max_allowed={max_allowed})")
            except Exception as e:
                logger.debug(f"build_allowed_mask skip (V={V}, max_allowed={max_allowed}): {e}")

        logger.info("Helper kernel precompilation finished")

    def compile(self, *, max_num_batched_tokens: int | None = None) -> None:
        """Compile the model for token/request bucket sizes.

        Drives the execution manager's bucketed compile over the current
        ``num_tokens_paddings`` and active sequence buckets so the hot path
        has zero JIT cost.

        Args:
            max_num_batched_tokens: Optional per-step token budget. When set,
                compilation is capped to the smallest token bucket that is at
                least ``max_num_batched_tokens``; otherwise every bucket up to
                ``max_num_tokens`` is compiled.

        Raises:
            ValueError: If ``max_num_batched_tokens`` is provided but not
                positive.

        Notes:
            - ``max_model_len`` controls the *sequence length* (context window).
            - ``max_num_batched_tokens`` controls the *per-step* token budget
              that the scheduler will emit in a single forward pass.

            When ``max_num_batched_tokens`` is provided, compilation is capped
            to the smallest token bucket >= that value (dramatically reducing
            startup time for long-context models).
        """
        logger.info("Starting eSurgeRunner compilation")
        num_tokens_paddings = list(self.num_tokens_paddings)
        if max_num_batched_tokens is not None:
            target = int(max_num_batched_tokens)
            if target <= 0:
                raise ValueError(f"max_num_batched_tokens must be positive, got {max_num_batched_tokens}")

            # Pick the smallest bucket >= target (keeps runtime bucket selection valid).
            cap = next((b for b in num_tokens_paddings if b >= target), num_tokens_paddings[-1])
            num_tokens_paddings = [b for b in num_tokens_paddings if b <= cap]

        logger.debug(
            f"Compiling for {len(num_tokens_paddings)} token padding sizes: {num_tokens_paddings[:5]}..."
            if len(num_tokens_paddings) > 5
            else f"Compiling for token padding sizes: {num_tokens_paddings}"
        )

        self.executor_manager.compile(
            num_tokens_paddings=num_tokens_paddings,
            num_reqs_max_model_len=self.num_reqs_max_model_len,
            max_pages_per_req=self.max_pages_per_req,
            max_num_reqs=self.max_num_reqs,
            metadata=self.metadata,
            num_reqs_paddings=self.active_num_seq_buckets,
            prune_infeasible_pairs=self._allow_sparse_window_packing,
        )

        self._precompile_vlm_vision_helpers(max_num_batched_tokens=num_tokens_paddings[-1])

        helper_prompt_buckets = [min(n, self.max_model_len) for n in num_tokens_paddings]
        if self.pipeline_plan.is_enabled:
            helper_prompt_buckets = sorted({helper_prompt_buckets[0], helper_prompt_buckets[-1]})

        self._precompile_jitted_helpers(
            reqs_padds=self.active_num_seq_buckets,
            prompt_len_buckets=helper_prompt_buckets,
            precompile_allowed_mask=False,
            allowed_max=4096,
        )

    def update_model_weights(
        self,
        model: EasyDeLBaseModule | None = None,
        *,
        graphdef=None,
        graphstate=None,
        graphother=None,
        reset_state: bool = True,
    ) -> None:
        """Update the runner's model weights/graphs and optionally reset state.

        Args:
            model: Optional EasyDeL model instance providing new weights. If
                omitted, graph components must be supplied explicitly.
            graphdef: Optional graphdef override.
            graphstate: Optional graphstate override.
            graphother: Optional graphother override.
            reset_state: When True (default) reinitializes internal buffers and
                cached requests to ensure the new weights are applied cleanly.

        Raises:
            RuntimeError: If active requests exist while reset_state is True.
        """
        if reset_state and self.requests:
            raise RuntimeError("Cannot update model weights while requests are active")

        if model is None:
            if graphdef is None:
                raise ValueError("graphdef must not be None when model is None")
            if graphstate is None:
                raise ValueError("graphstate must not be None when model is None")
            if graphother is None:
                raise ValueError("graphother must not be None when model is None")
            model = spx.bind(graphdef, graphstate.overlay(graphother))

        model = model.esurge_compatible_model
        graphdef = model.graphdef
        self.model = model
        self._vlm_image_features_jit = None
        self._vlm_video_features_jit = None
        self._vlm_vision_jit_disabled = False

        self.executor_manager.update_graphs(
            model=model,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
        )

        if reset_state:
            self._setup_variables()

    def release_model_state(self, *, clear_compiled_cache: bool = False) -> None:
        """Drop model/graph references held by the runner to free memory.

        This keeps the runner object reusable, but it requires a later
        `update_model_weights(...)` call before executing new generation steps.

        Args:
            clear_compiled_cache: Whether to clear compiled model/sampler caches.

        Raises:
            RuntimeError: If active requests exist.
        """
        if self.requests:
            raise RuntimeError("Cannot release model state while requests are active")

        self.reset_state()

        if clear_compiled_cache:
            self.executor_manager.clear_cache()

        # Drop strong references to model and device-resident graph trees.
        self.model = None
        self._vlm_image_features_jit = None
        self._vlm_video_features_jit = None
        self._vlm_vision_jit_disabled = False
        self.executor_manager.model = None
        self.executor_manager.graphstate = None
        self.executor_manager.graphother = None
        self.executor_manager._model_executor.model = None
        self.executor_manager._model_executor.clear_runtime_graph_args()
        self.executor_manager._sampler_executor.model = None

    def destroy_kv_cache(self) -> None:
        """Drop the executor-manager's KV-pages reference to free HBM.

        Called from :meth:`eSurge.pause` when ``destroy_pages_on_pause``
        is enabled. Does not zero the underlying device buffers — Python
        garbage collection of the cache object releases the HBM as soon
        as no other reference remains. Counterpart of
        :meth:`initialize_kv_cache`, which reallocates a fresh cache on
        resume.
        """
        logger.info("Destroying eSurgeRunner ragged KV cache pages")
        self.executor_manager.kv_pages = None

    def initialize_kv_cache(self) -> None:
        """Allocate the operations cache when the executor manager has none.

        Idempotent: when ``executor_manager.kv_pages`` is already set,
        returns immediately so resuming an already-running engine doesn't
        leak a second allocation. Otherwise builds the right
        ``Quantizer`` (TurboQuant gets a no-config quantizer; everything
        else gets the model's standard kv-quant config) and delegates to
        the bounded-retry path
        :meth:`ExecutionManager._init_operations_cache_with_retry`, which
        will shrink the page pool on PP HBM-OOM. Called by both
        :meth:`eSurge.initiate` (fresh start) and :meth:`eSurge.resume`
        (after a pause that destroyed pages).
        """

        if self.executor_manager.kv_pages is not None:
            logger.debug("KV cache already initialized; skipping reallocation")
            return

        logger.info("Reinitializing eSurgeRunner ragged KV cache pages")
        text_config = self.model.config.get_text_config()
        kv_quant_cfg = text_config.kv_cache_quantization_config
        # TurboQuant handles compression internally; skip standard quantizer
        _is_turboquant = isinstance(kv_quant_cfg, TurboQuantConfig)
        if _is_turboquant:
            quantizer = self.model._quant_class(quantization_config=None)
        else:
            quantizer = self.model._quant_class(quantization_config=kv_quant_cfg)

        self.executor_manager.kv_pages = self.executor_manager._init_operations_cache_with_retry(
            quantizer=quantizer,
            masking_details=getattr(text_config, "get_mask_details", lambda: None)(),
        )
        return

    # VLM prefill passthroughs
    # The VLM prefill machinery lives on ``self.vlm`` (:class:`VlmPrefillHelper`).
    # The historical runner method/attribute names are kept as thin delegators
    # and properties so _execute_model_impl, compile(), weight-update paths, and
    # external tests are untouched.

    @property
    def compile_vision_encoder(self) -> bool:
        """Whether the bucketed vision JIT path is enabled (see :class:`VlmPrefillHelper`)."""
        return self.vlm.compile_vision_encoder

    @property
    def vision_patch_buckets(self) -> list[int] | None:
        """Raw patch-count buckets for vision precompile (see :class:`VlmPrefillHelper`)."""
        return self.vlm.vision_patch_buckets

    @property
    def _vlm_image_features_jit(self) -> typing.Callable | None:
        """Cached image-features JIT closure (see :class:`VlmPrefillHelper`)."""
        return self.vlm.image_features_jit

    @_vlm_image_features_jit.setter
    def _vlm_image_features_jit(self, value: typing.Callable | None) -> None:
        self.vlm.image_features_jit = value

    @property
    def _vlm_video_features_jit(self) -> typing.Callable | None:
        """Cached video-features JIT closure (see :class:`VlmPrefillHelper`)."""
        return self.vlm.video_features_jit

    @_vlm_video_features_jit.setter
    def _vlm_video_features_jit(self, value: typing.Callable | None) -> None:
        self.vlm.video_features_jit = value

    @property
    def _vlm_vision_jit_disabled(self) -> bool:
        """Sticky vision-JIT fallback flag (see :class:`VlmPrefillHelper`)."""
        return self.vlm.vision_jit_disabled

    @_vlm_vision_jit_disabled.setter
    def _vlm_vision_jit_disabled(self, value: bool) -> None:
        self.vlm.vision_jit_disabled = value

    @property
    def _vlm_cpu_buffers(self):
        """Host-side VLM scratch buffers keyed by bucket (see :class:`VlmPrefillHelper`)."""
        return self.vlm.cpu_buffers

    @staticmethod
    def _static_grid_thw(grid: np.ndarray | jax.Array | tuple | list | None) -> tuple[tuple[int, int, int], ...] | None:
        """Convert a processor grid into a hashable static JIT argument.

        Delegates to :meth:`VlmPrefillHelper.static_grid_thw`.
        """
        return VlmPrefillHelper.static_grid_thw(grid)

    @staticmethod
    def _max_grid_size(grid: tuple[tuple[int, int, int], ...] | None) -> int | None:
        """Return the static max spatial grid size used by Qwen-style vision towers.

        Delegates to :meth:`VlmPrefillHelper.max_grid_size`.
        """
        return VlmPrefillHelper.max_grid_size(grid)

    @staticmethod
    def _block_tree_until_ready(value: typing.Any) -> None:
        """Block until every device-array leaf in ``value`` is ready.

        Delegates to :meth:`VlmPrefillHelper.block_tree_until_ready`.
        """
        VlmPrefillHelper.block_tree_until_ready(value)

    def _vlm_uses_deepstack_visuals(self) -> bool:
        """Delegates to :meth:`VlmPrefillHelper.uses_deepstack_visuals`."""
        return self.vlm.uses_deepstack_visuals()

    def _vision_patch_input_dim(self) -> int | None:
        """Delegates to :meth:`VlmPrefillHelper.vision_patch_input_dim`."""
        return self.vlm.vision_patch_input_dim()

    def _vision_spatial_merge_size(self) -> int:
        """Delegates to :meth:`VlmPrefillHelper.vision_spatial_merge_size`."""
        return self.vlm.vision_spatial_merge_size()

    def _vision_dtype(self) -> jnp.dtype:
        """Delegates to :meth:`VlmPrefillHelper.vision_dtype`."""
        return self.vlm.vision_dtype()

    def _vision_grid_for_patch_bucket(self, raw_patches: int) -> tuple[int, tuple[int, int, int]]:
        """Delegates to :meth:`VlmPrefillHelper.vision_grid_for_patch_bucket`."""
        return self.vlm.vision_grid_for_patch_bucket(raw_patches)

    def _vision_patch_buckets_for_compile(self, *, max_num_batched_tokens: int | None) -> list[int]:
        """Delegates to :meth:`VlmPrefillHelper.vision_patch_buckets_for_compile`."""
        return self.vlm.vision_patch_buckets_for_compile(max_num_batched_tokens=max_num_batched_tokens)

    def _get_vlm_image_features_jit(self) -> typing.Callable:
        """Delegates to :meth:`VlmPrefillHelper.get_image_features_jit`."""
        return self.vlm.get_image_features_jit()

    def _get_vlm_video_features_jit(self) -> typing.Callable:
        """Delegates to :meth:`VlmPrefillHelper.get_video_features_jit`."""
        return self.vlm.get_video_features_jit()

    def _compute_embedding_with_info_single_pass(
        self,
        input_ids: jax.Array,
        embed_kwargs: dict[str, typing.Any],
    ) -> tuple[jax.Array, typing.Any]:
        """Compute VLM embeddings while avoiding wrapper-level duplicate vision work.

        Delegates to :meth:`VlmPrefillHelper.compute_embedding_with_info_single_pass`.
        """
        return self.vlm.compute_embedding_with_info_single_pass(input_ids, embed_kwargs)

    def _compiled_vision_embed_kwargs(self, req_state: CachedRequestState) -> dict[str, typing.Any] | None:
        """Delegates to :meth:`VlmPrefillHelper.compiled_vision_embed_kwargs`."""
        return self.vlm.compiled_vision_embed_kwargs(req_state)

    def _compute_vlm_prefill_with_compiled_vision(
        self,
        req_state: CachedRequestState,
        input_ids: jax.Array,
        attention_mask: jax.Array,
    ) -> tuple[jax.Array, typing.Any] | None:
        """Delegates to :meth:`VlmPrefillHelper.compute_prefill_with_compiled_vision`."""
        return self.vlm.compute_prefill_with_compiled_vision(
            req_state=req_state,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def _precompile_vlm_vision_helpers(self, *, max_num_batched_tokens: int | None) -> None:
        """Delegates to :meth:`VlmPrefillHelper.precompile_vision_helpers`."""
        self.vlm.precompile_vision_helpers(max_num_batched_tokens=max_num_batched_tokens)

    def _precompute_vlm_prefill(self, req_state: CachedRequestState) -> None:
        """Precompute prompt embeddings (+ optional mRoPE indices) for VLM requests.

        Delegates to :meth:`VlmPrefillHelper.precompute_prefill`.
        """
        self.vlm.precompute_prefill(req_state)

    def _update_states(self, scheduler_output: SchedulerOutput) -> bool:
        """Update internal states based on scheduler output.

        Delegates to :meth:`SequenceStateSync.update_states`.

        Args:
            scheduler_output: Contains request scheduling decisions (finished,
                new, cached requests and per-request token counts).

        Returns:
            True if state changed (requests added/removed), indicating
            potential buffer reorganization. False if no changes occurred.
        """
        return self.state_sync.update_states(scheduler_output)

    def _modify_prev_results(self, pre_results: AsyncPreResults | None = None) -> None:
        """Apply previous iteration's tokens to sequence buffer.

        Delegates to :meth:`SequenceStateSync.modify_prev_results`, defaulting
        to the runner's pending async payload when ``pre_results`` is omitted.
        """
        if pre_results is None:
            pre_results = self._pre_async_results
        self.state_sync.modify_prev_results(pre_results)

    def _finalize_async_scheduler_runner_state(
        self,
        sampled_token_ids: list[list[int]],
        *,
        request_seq_lens: list[tuple[int, int, CachedRequestState, int]],
        expected_pre_results: AsyncPreResults | None = None,
    ) -> None:
        """Repair runner-side placeholders after async scheduler output drains.

        Delegates the token repair to
        :meth:`SequenceStateSync.finalize_async_runner_state`, then clears the
        runner's pending async payload. ``expected_pre_results`` lets the drain
        path clear only the payload it finalized: in the PP device-handoff
        path, step ``N+1`` may have already installed its own async payload by
        the time step ``N`` drains; clearing by identity avoids deleting that
        newer payload.
        """
        SequenceStateSync.finalize_async_runner_state(
            sampled_token_ids,
            request_seq_lens=request_seq_lens,
            sequence_buffer=self.sequence_buffer,
            requests=self.requests,
            max_model_len=self.max_model_len,
        )
        if expected_pre_results is None or self._pre_async_results is expected_pre_results:
            self._pre_async_results = None

    def _model_uses_vlm_inputs(self) -> bool:
        """Return whether the loaded model has multimodal prefill side inputs."""
        cfg = getattr(self.model, "config", None)
        task_type = getattr(self.model, "_task_type", None)
        return task_type == "image-text-to-text" or (
            cfg is not None
            and (getattr(cfg, "image_token_id", None) is not None or getattr(cfg, "video_token_id", None) is not None)
            and callable(getattr(self.model, "get_image_features", None))
        )

    def _has_default_sampling_penalties(self) -> bool:
        """Return whether device-token handoff can ignore penalty state updates.

        Async placeholder handoff patches only the next input token. Frequency,
        presence, and repetition penalties are computed from CPU-side token
        history during metadata preparation, so requests with non-default
        penalties must keep the older repair-before-dispatch path until those
        penalty buffers are also made device-resident.
        """
        return (
            not bool(np.any(self.sequence_buffer.frequency_penalties != 0.0))
            and not bool(np.any(self.sequence_buffer.presence_penalties != 0.0))
            and not bool(np.any(self.sequence_buffer.repetition_penalties != 1.0))
        )

    def can_dispatch_next_before_async_drain(self, scheduler_output: SchedulerOutput) -> bool:
        """Return whether the next async decode step may launch before drain.

        The async scheduler advances decode with optimistic output placeholders.
        If this returns ``True``, the lifecycle loop can schedule and dispatch
        step ``N+1`` before host-materializing step ``N``. The runner then
        replaces the placeholder input in ``N+1`` from step ``N``'s device
        sampled-token array via :class:`DeviceInputTokenHandoff`.

        This is deliberately a capability gate, not a pipeline-parallelism
        check. It applies to both SPMD/TP and PP executions, but only when all
        correctness-sensitive features that currently depend on CPU token
        history are disabled.
        """
        if not self.async_scheduling or not scheduler_output.async_scheduling:
            return False
        if int(scheduler_output.total_num_scheduled_tokens or 0) <= 0:
            return False
        if any(int(n) != 1 for n in scheduler_output.num_scheduled_tokens.values()):
            return False
        if bool(getattr(scheduler_output, "pending_structured_output_tokens", False)):
            return False
        if self._model_uses_vlm_inputs():
            return False
        return self._has_default_sampling_penalties()

    def _can_delay_async_result_repair(
        self,
        *,
        scheduler_output: SchedulerOutput,
        return_async_output: bool,
        is_vlm_model: bool,
        frequency_penalties_cpu: np.ndarray,
        presence_penalties_cpu: np.ndarray,
        repetition_penalties_cpu: np.ndarray,
    ) -> bool:
        """Return whether previous async tokens may be repaired after dispatch.

        The delayed path is a launch-path optimization: it lets the next decode
        step consume previous sampled tokens from device arrays, then repairs
        CPU sequence state after that step has been queued. It is only
        profitable when the caller is already using the deferred-output path;
        the synchronous async-scheduler loop has already materialized the token
        for ``scheduler.update_from_output()``, so adding a scatter there just
        adds work. Keep the gate conservative so correctness-sensitive features
        continue using the old host-repair-before-dispatch path.
        """
        if not return_async_output:
            return False
        if not scheduler_output.async_scheduling:
            return False
        if int(scheduler_output.total_num_scheduled_tokens or 0) <= 0:
            return False
        if any(int(n) != 1 for n in scheduler_output.num_scheduled_tokens.values()):
            return False
        if bool(getattr(scheduler_output, "pending_structured_output_tokens", False)):
            return False
        if is_vlm_model:
            return False
        if bool(np.any(frequency_penalties_cpu != 0.0)):
            return False
        if bool(np.any(presence_penalties_cpu != 0.0)):
            return False
        if bool(np.any(repetition_penalties_cpu != 1.0)):
            return False
        return True

    def _build_device_token_handoff(
        self,
        *,
        pre_results: AsyncPreResults,
        req_ids_window: list[str | None],
        scheduled_list: list[int],
        window_row_indices: np.ndarray,
        num_tokens_static: int,
    ) -> DeviceInputTokenHandoff | None:
        """Build a device-side patch for decode placeholders in one window.

        The current batch preparer reads token ids from the CPU
        ``sequence_buffer``. When async scheduling inserted a placeholder at the
        previous step, that CPU slot still contains ``0`` until
        :meth:`_modify_prev_results` materializes the sampled token. For pure
        decode rows whose next input is exactly that placeholder position, this
        helper gathers the previous device sampled-token scalar and records the
        flattened input offset that must be patched.

        Returning ``None`` means the caller must use the old path and repair CPU
        state before building/launching the next step.
        """
        sampled_by_req: dict[str, tuple[int, jax.Array, int]] = {}
        for window_idx, window in enumerate(pre_results.windows):
            for row_pos, req_id, is_valid in zip(window.row_positions, window.req_ids, window.valid_mask, strict=False):
                if is_valid:
                    sampled_by_req[str(req_id)] = (window_idx, window.sampled_token_ids, int(row_pos))

        use_spmd_dp = self._uses_spmd_dp()
        if use_spmd_dp:
            dp_size = max(1, int(getattr(self.metadata, "data_parallel_size", 1) or 1))
            if int(num_tokens_static) % dp_size != 0:
                return None
            if int(self.num_reqs_max_model_len) % dp_size != 0:
                return None
            tokens_per_rank = int(num_tokens_static) // dp_size
            rows_per_rank = int(self.num_reqs_max_model_len) // dp_size
            rank_used_tokens = [0] * dp_size
        else:
            dp_size = 1
            tokens_per_rank = int(num_tokens_static)
            rows_per_rank = int(self.num_reqs_max_model_len)
            rank_used_tokens = [0]

        patch_positions: list[int] = []
        patch_token_sources: list[tuple[jax.Array, int]] = []
        patch_sources: list[tuple[int, int]] = []
        flat_offset = 0
        for local_row, rid in enumerate(req_ids_window):
            scheduled = int(scheduled_list[local_row])
            if scheduled <= 0:
                continue
            global_row = int(window_row_indices[local_row])
            if use_spmd_dp:
                rank = min(max(global_row, 0) // rows_per_rank, dp_size - 1)
                packed_row_offset = rank * tokens_per_rank + int(rank_used_tokens[rank])
                if packed_row_offset + scheduled > (rank + 1) * tokens_per_rank:
                    return None
            else:
                rank = 0
                packed_row_offset = flat_offset
            if rid is None:
                if use_spmd_dp:
                    rank_used_tokens[rank] += scheduled
                flat_offset += scheduled
                continue

            start_tok = int(self.sequence_buffer.num_computed_tokens[global_row])
            placeholder_pos = int(self.sequence_buffer.num_tokens_no_spec[global_row]) - 1
            touches_placeholder = start_tok <= placeholder_pos < start_tok + scheduled
            if not touches_placeholder:
                if use_spmd_dp:
                    rank_used_tokens[rank] += scheduled
                flat_offset += scheduled
                continue
            if scheduled != 1 or start_tok != placeholder_pos:
                return None
            sampled = sampled_by_req.get(str(rid))
            if sampled is None:
                return None

            sampled_window_idx, sampled_tokens, sampled_row_pos = sampled
            patch_positions.append(packed_row_offset + (placeholder_pos - start_tok))
            patch_token_sources.append((sampled_tokens, sampled_row_pos))
            patch_sources.append((sampled_window_idx, sampled_row_pos))
            if use_spmd_dp:
                rank_used_tokens[rank] += scheduled
            flat_offset += scheduled

        if not patch_token_sources:
            return None

        max_handoff = int(getattr(self.executor_manager, "max_num_reqs", len(patch_positions)) or len(patch_positions))
        if len(patch_positions) > max_handoff:
            return None

        count = self._get_handoff_scalar(len(patch_positions))
        offset = self._get_handoff_scalar(0)
        fast_dense_handoff = (
            patch_positions == list(range(len(patch_positions)))
            and patch_sources == [(patch_sources[0][0], idx) for idx in range(len(patch_sources))]
            and int(pre_results.windows[patch_sources[0][0]].sampled_token_ids.shape[0]) >= max_handoff
        )
        if fast_dense_handoff:
            input_positions = self._get_handoff_positions(max_handoff)
            token_ids = pre_results.windows[patch_sources[0][0]].sampled_token_ids
        else:
            input_positions_np = np.asarray(patch_positions, dtype=np.int32)
            if input_positions_np.shape[0] < max_handoff:
                input_positions_np = np.pad(input_positions_np, (0, max_handoff - int(input_positions_np.shape[0])))
            input_positions = jax.device_put(input_positions_np, self.executor_manager._empty_sharding)
            token_ids = jnp.stack(
                [
                    jnp.asarray(sampled_tokens[row_pos], dtype=jnp.int32)
                    for sampled_tokens, row_pos in patch_token_sources
                ]
            )
            if token_ids.shape[0] < max_handoff:
                token_ids = jnp.pad(token_ids, ((0, max_handoff - int(token_ids.shape[0])),))
            token_ids = jax.device_put(token_ids, self.executor_manager._empty_sharding)
        return DeviceInputTokenHandoff(input_positions=input_positions, token_ids=token_ids, count=count, offset=offset)

    def _get_handoff_positions(self, size: int) -> jax.Array:
        """Return cached ``[0, size)`` device positions for dense token handoff."""
        cached = self._handoff_positions_cache.get(size)
        if cached is not None:
            return cached
        positions = np.arange(size, dtype=np.int32)
        cached = jax.device_put(positions, self.executor_manager._empty_sharding)
        self._handoff_positions_cache[size] = cached
        return cached

    def _get_handoff_scalar(self, value: int) -> jax.Array:
        """Return a cached int32 device scalar used by token handoff metadata."""
        cached = self._handoff_scalar_cache.get(value)
        if cached is not None:
            return cached
        cached = jax.device_put(np.asarray(value, dtype=np.int32), self.executor_manager._empty_sharding)
        self._handoff_scalar_cache[value] = cached
        return cached

    def _update_placeholder(
        self,
        discard_sampled_tokens_req_indices: list[int],
        request_seq_lens: list[tuple[int, int, CachedRequestState, int]],
    ) -> dict[str, int]:
        """Set placeholders for tokens not yet generated.

        When async scheduling is enabled, this method is called after the
        forward pass to set placeholder tokens (0) for requests that will
        generate tokens. The actual tokens will be filled in during the
        next iteration via _modify_prev_results().

        Args:
            discard_sampled_tokens_req_indices: Indices of requests whose
                tokens should be discarded (e.g., partial prefill).
            request_seq_lens: List of (out_idx, seq_row_idx, req_state,
                seq_len) tuples for requests that generated tokens.

        Returns:
            Mapping from request ID to index for placeholder replacement.

        Note:
            This method updates num_tokens_no_spec and num_tokens in the
            sequence buffer, and appends placeholder (0) to output_token_ids.
        """
        placeholder_req_id_to_index: dict[str, int] = {}
        discard_set = set(discard_sampled_tokens_req_indices)

        for out_idx, seq_row_idx, req_state, placeholder_idx in request_seq_lens:
            if out_idx in discard_set:
                continue

            # Honor the placeholder position recorded at schedule time
            # (``num_computed_tokens + scheduled``). Re-reading
            # ``num_tokens_no_spec`` here would re-introduce the stale-counter
            # drift after a synchronously finalized step.
            start_idx = int(placeholder_idx)
            end_idx = start_idx + 1  # Assume 1 token (no spec decode yet)

            if end_idx > self.max_model_len:
                raise ValueError(
                    f"Sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: {self.max_model_len}"
                )

            # Update buffer state
            self.sequence_buffer.num_tokens_no_spec[seq_row_idx] = end_idx
            self.sequence_buffer.num_tokens[seq_row_idx] = end_idx

            # Add placeholder (0) to output
            req_state.output_token_ids.extend([0])
            placeholder_req_id_to_index[req_state.req_id] = seq_row_idx

        return placeholder_req_id_to_index

    def _reorder_decode_first(self, scheduler_output: SchedulerOutput) -> None:
        """Reorder active requests so decode requests are placed first.

        Delegates to :meth:`SequenceStateSync.reorder_decode_first`.
        """
        self.state_sync.reorder_decode_first(scheduler_output)

    def _reorder_decode_first_per_shard(
        self,
        scheduler_output: SchedulerOutput,
        dp_size: int,
    ) -> None:
        """Reorder decode requests first within each DP shard's row range.

        Delegates to :meth:`SequenceStateSync.reorder_decode_first_per_shard`.
        """
        self.state_sync.reorder_decode_first_per_shard(scheduler_output, dp_size)

    def _execute_model_impl(
        self,
        scheduler_output: SchedulerOutput,
        *,
        return_async_output: bool = False,
    ) -> ModelRunnerOutput | _AsyncExecutionHandle:
        """Execute the model on scheduled requests.

        Main entry point for model execution. Processes all scheduled requests
        in batches, handling state updates, input preparation, model execution,
        and token processing.

        The method handles:
        1. State synchronization with scheduler
        2. Batch-wise processing of requests
        3. Token generation and sampling
        4. Buffer updates and metrics logging

        Args:
            scheduler_output: Output from the scheduler containing:
                - Requests to process
                - Tokens to generate per request
                - Finished/new/cached request information

        Returns:
            ModelRunnerOutput or _AsyncExecutionHandle. The async handle is used
            by overlap execution to defer the host block while preserving
            same-thread TPU dispatch.

            ModelRunnerOutput contains:
                - req_ids: List of processed request IDs
                - sampled_token_ids: Generated tokens per request
                - logprobs: Log probabilities (if requested)
                - Timing and debugging information

        Note:
            The method processes requests in batches when they exceed
            the maximum model length, ensuring all requests are handled
            efficiently without exceeding memory constraints.
        """
        execution_start_time = time.time()

        updating_states_start = time.time()
        layout_version_before = self.sequence_buffer.layout_version
        prev_recurrent_row_owner = (
            list(self.sequence_buffer.req_ids) if self._plain_recurrent_row_sync_enabled() else None
        )
        self._update_states(scheduler_output)
        self.spec.on_states_applied(scheduler_output)
        updating_states_time = time.time() - updating_states_start

        # Apply previous async results if available. For safe PP decode windows,
        # this repair is delayed until after the next device dispatch is queued:
        # the next input token is patched from the previous device token directly.
        prev_async_start = time.time()
        pending_pre_async_results = self._pre_async_results
        self._pre_async_results = None
        pre_async_repair_pending = pending_pre_async_results is not None
        prev_async_time = time.time() - prev_async_start

        # Align ordering with TPU runner: decode requests first.
        dp_size = int(getattr(self.metadata, "data_parallel_size", 1) or 1)
        if self.sequence_buffer.num_reqs > 1:
            if dp_size <= 1:
                self._reorder_decode_first(scheduler_output)
            else:
                self._reorder_decode_first_per_shard(scheduler_output, dp_size)

        if self.sequence_buffer.layout_version != layout_version_before:
            self.executor_manager.invalidate_sampler_penalty_state(
                self.sequence_buffer.token_ids,
                self.sequence_buffer.num_tokens,
            )
            # Keep the plain-recurrent conv/GDR/SSM device state (indexed by
            # physical buffer row) in lockstep with condense/reorder row moves.
            self._sync_recurrent_rows(prev_recurrent_row_owner)

        if not scheduler_output.total_num_scheduled_tokens:
            if pending_pre_async_results is not None and pre_async_repair_pending:
                prev_async_start = time.time()
                self._modify_prev_results(pending_pre_async_results)
                pre_async_repair_pending = False
                prev_async_time += time.time() - prev_async_start
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                req_id_to_row_index={},
                sampled_token_ids=[],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
                finished_sending=None,
                finished_recving=None,
                num_nans_in_logits=None,
            )

        can_defer_async_output = (
            self.drafter is None
            and bool(scheduler_output.async_scheduling)
            and self.can_dispatch_next_before_async_drain(scheduler_output)
        )
        needs_async_output = return_async_output or can_defer_async_output
        if self.drafter is not None and return_async_output:
            raise NotImplementedError("eSurge runner-native speculative decoding is synchronous-only for now.")
        start_index = 0
        total_step_time = 0.0
        total_post_proc_time = 0.0

        req_ids_all: list[str] = []
        sampled_token_ids_all: list[list[int]] = []
        spec_token_ids_all: list[list[int]] | None = [] if self.drafter is not None else None
        accepted_spec_tokens_by_req: dict[str, int] | None = {} if self.drafter is not None else None
        hidden_states_by_req: dict[str, typing.Any] | None = {} if self.drafter is not None else None
        token_logprobs: dict[str, float] = {}
        async_windows: list[AsyncWindowResult] = []
        sync_finalize_entries: list[tuple[CachedRequestState | None, int | None, int | None]] = []

        # Window-level perf aggregation (a single scheduler step can span multiple windows).
        num_windows = 0
        total_exec_time = 0.0
        total_sample_time = 0.0
        total_prep_time = 0.0
        total_prep_host_time = 0.0
        total_prep_put_time = 0.0
        total_prep_extra_put_time = 0.0
        total_prep_batch_metadata_time = 0.0
        total_prep_handoff_time = 0.0
        total_prep_sampler_window_time = 0.0
        total_prep_ensure_variants_time = 0.0
        total_prep_pack_inputs_time = 0.0
        total_execute_overhead_time = 0.0
        total_model_enqueue_time = 0.0
        total_sampler_enqueue_time = 0.0
        total_greedy_argmax_time = 0.0
        total_greedy_argmax_fastpath = 0
        total_logits_wait_time = 0.0
        total_exec_enqueue_time = 0.0
        total_exec_wait_time = 0.0
        total_sampler_wait_time = 0.0
        total_pp_stage_dispatch_time = 0.0
        total_pp_queue_wait_time = 0.0
        total_pp_stage_launches = 0
        total_pp_stage_compute_time = 0.0
        total_pp_stage_max_time = 0.0
        total_pp_prepare_time = 0.0
        total_pp_submit_time = 0.0
        total_pp_assemble_time = 0.0
        total_pp_backbone_time = 0.0
        total_pp_combine_time = 0.0
        total_pp_lm_head_time = 0.0
        total_pp_sampler_enqueue_time = 0.0
        total_pp_microbatch_scratch_time = 0.0
        total_pp_microbatch_metadata_time = 0.0
        total_pp_microbatch_handoff_time = 0.0
        total_pp_sampler_window_time = 0.0
        total_pp_ensure_variants_time = 0.0
        pp_stage_times_by_index: dict[int, float] = {}
        pp_stage_submit_times_by_index: dict[int, float] = {}
        pp_stage_assemble_times_by_index: dict[int, float] = {}
        pp_stage_execute_times_by_index: dict[int, float] = {}
        total_runner_host_time = 0.0
        total_async_copy_enqueue_time = 0.0
        total_token_materialize_time = 0.0
        total_spec_project_time = 0.0
        total_spec_argmax_sync_time = 0.0
        total_spec_meta_time = 0.0
        total_spec_emit_write_time = 0.0
        total_spec_draft_time = 0.0
        self._spec_suffix_time_acc = 0.0
        self._spec_replay_time_acc = 0.0
        total_spec_commit_time = 0.0
        # Greedy inline-MTP requests collected during the window loop(s) for ONE
        # post-loop batched draft pass, instead of ``N`` per-request drafts. Each
        # entry: (req_id, req_idx, seed_token, seed_position, seed_hidden, req_state,
        # known_len, spec_idx). See ``DrafterSpeculation.draft_next_batched``.
        pending_batched_drafts: list[tuple] = []
        # Only engage the batched pool draft when >= 2 spec-active requests are being
        # verified this step. Batching a single request wins no throughput (a pooled
        # [1, ...] forward is no cheaper than one row-targeted forward) but moves the
        # draft forward past the window loop, reordering it relative to the recurrent
        # replay/commit; on a near-tie greedy argmax that XLA:CPU execution-order
        # change can flip the verify's next token. Below the threshold each request
        # drafts in-loop through the exact per-request path, so single-request serving
        # stays bit-identical to the non-batched drafter. (For >= 2 the pooled forward
        # is used; its batched matmul differs from N separate matmuls by ~1 ULP, the
        # same near-tie sensitivity any batched-decode inference engine has.)
        step_batch_candidate_count = 0
        if self.drafter is not None and scheduler_output.scheduled_spec_decode_tokens:
            for _spec_rid, _spec_toks in scheduler_output.scheduled_spec_decode_tokens.items():
                if _spec_toks and all(int(_t) >= 0 for _t in _spec_toks) and self.spec.can_batch_draft(
                    self.requests.get(_spec_rid)
                ):
                    step_batch_candidate_count += 1
        step_allows_batch = step_batch_candidate_count >= 2
        token_buckets_used: set[int] = set()
        req_buckets_used: set[int] = set()
        request_seq_lens: list[tuple[int, int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices: list[int] = []

        is_vlm_model = self._model_uses_vlm_inputs()
        if self._uses_spmd_dp():
            is_vlm_model = False
        uses_mrope_model = is_vlm_model and model_uses_mrope(self.model)

        while start_index < self.sequence_buffer.num_slots:
            host_start = time.time()
            num_reqs_total = self.sequence_buffer.num_slots
            window_stop_index = min(num_reqs_total, start_index + self.num_reqs_max_model_len)
            (
                window_row_indices,
                req_ids_window,
                scheduled_list,
                next_start_index,
                packed_window_rows,
            ) = self._collect_schedulable_window_rows(
                start_index=start_index,
                stop_index=window_stop_index,
                scheduled_tokens_by_req=scheduler_output.num_scheduled_tokens,
                allow_sparse_packing=self._allow_sparse_window_packing and not scheduler_output.async_scheduling,
            )
            num_reqs = len(scheduled_list)
            if num_reqs == 0:
                start_index = next_start_index
                continue

            original_scheduled_list = [int(n) for n in scheduled_list]
            model_scheduled_list = list(original_scheduled_list)
            # SP3: intentionally always empty. No code path populates
            # `sequential_greedy_spec_rows`, so the sequential-greedy spec fast
            # path it guards below (`if int(row_pos) in sequential_greedy_spec_rows
            # and len(scheduled_spec_tokens) == 1`) is dead. It is kept, unrevived,
            # as the seam for a future one-token sequential-greedy verify: do NOT
            # populate it without also validating that fast path end-to-end.
            sequential_greedy_spec_rows: dict[int, int] = {}

            total_scheduled = sum(model_scheduled_list)
            token_bucket_target = total_scheduled
            if self._uses_spmd_dp():
                dp_size = max(1, int(getattr(self.metadata, "data_parallel_size", 1) or 1))
                rows_per_rank = int(self.num_reqs_max_model_len) // dp_size
                rank_scheduled_tokens = [0] * dp_size
                for row_pos, scheduled_tokens in enumerate(model_scheduled_list):
                    global_row = int(window_row_indices[row_pos])
                    rank = min(max(global_row, 0) // rows_per_rank, dp_size - 1)
                    rank_scheduled_tokens[rank] += int(scheduled_tokens)
                peak_rank_scheduled = max(rank_scheduled_tokens) if rank_scheduled_tokens else 0
                token_bucket_target = peak_rank_scheduled * dp_size
            idx = bisect_left(self.num_tokens_paddings, token_bucket_target)
            if idx >= len(self.num_tokens_paddings):
                idx = len(self.num_tokens_paddings) - 1
            num_tokens_static = int(self.num_tokens_paddings[idx])

            # Select optimal bucket for current batch size
            # This determines which compiled function to use
            current_bucket = self._get_current_bucket(num_reqs)
            padded_num_reqs = current_bucket  # Use bucket size for compilation lookup

            scheduled_full_cpu = self._scheduled_full_cpu
            active_mask_full_cpu = self._active_mask_full_cpu
            req_num_tokens_np = self._req_num_tokens_cpu
            window_row_indices_cpu = self._window_row_indices_cpu
            recurrent_slot_indices_cpu = self._window_recurrent_slot_indices_cpu
            if num_reqs > 0:
                # Keep scheduled and active_mask as CPU arrays
                scheduled_full_cpu = self._scheduled_full_cpu
                scheduled_full_cpu.fill(0)
                scheduled_full_cpu[: len(model_scheduled_list)] = model_scheduled_list

                # Packed view of the per-request target lengths for the current window.
                # Avoid per-step dict lookups; SequenceBuffer keeps this aligned with its ordering.
                req_num_tokens_np.fill(0)
                req_num_tokens_np[:num_reqs] = self.sequence_buffer.num_tokens[window_row_indices]
                for _row_pos in sequential_greedy_spec_rows:
                    req_num_tokens_np[_row_pos] = int(
                        self.sequence_buffer.num_computed_tokens[int(window_row_indices[_row_pos])]
                    ) + int(model_scheduled_list[_row_pos])

                active_mask_full_cpu = self._active_mask_full_cpu
                active_mask_full_cpu.fill(False)
                for i, rid in enumerate(req_ids_window):
                    if rid is not None:
                        active_mask_full_cpu[i] = True

                window_row_indices_cpu.fill(0)
                window_row_indices_cpu[:num_reqs] = window_row_indices

                recurrent_slot_indices_cpu.fill(0)
                for row_pos, rid in enumerate(req_ids_window):
                    if rid is None:
                        continue
                    slot = self._recurrent_slot_by_req.get(str(rid))
                    if slot is None:
                        if self._uses_spmd_dp():
                            rank = self._request_dp_rank_by_req.get(str(rid))
                            if rank is None:
                                rows_per_rank = self._recurrent_rows_per_dp_rank()
                                rank = min(max(int(window_row_indices[row_pos]), 0) // rows_per_rank, dp_size - 1)
                            slot = self._assign_recurrent_slot(str(rid), int(rank))
                        elif self._slot_indexed_state:
                            slot = self._assign_recurrent_slot(str(rid), None)
                        else:
                            slot = int(window_row_indices[row_pos])
                    recurrent_slot_indices_cpu[row_pos] = int(slot)

                if host_payload_broadcast_needed():
                    req_num_tokens_np = multihost_utils.broadcast_one_to_all(req_num_tokens_np)

            mrope_position_ids_cpu: np.ndarray | None = None
            prefill_embeds_cpu: np.ndarray | None = None
            prefill_embeds_mask_cpu: np.ndarray | None = None
            visual_pos_masks_cpu = None
            deepstack_visual_embeds_cpu = None
            if is_vlm_model:
                # Precompute per-request VLM prompt embeddings outside the compiled step.
                for rid in req_ids_window:
                    if rid is None:
                        continue
                    req_state = self.requests.get(rid)
                    if req_state is None:
                        continue
                    if req_state.has_vision and not req_state.vision_processed:
                        self._precompute_vlm_prefill(req_state)

                (
                    prefill_embeds_cpu,
                    prefill_embeds_mask_cpu,
                    mrope_position_ids_cpu,
                    visual_pos_masks_cpu,
                    deepstack_visual_embeds_cpu,
                ) = self._get_vlm_cpu_buffers(
                    num_tokens_static=num_tokens_static,
                    uses_mrope_model=uses_mrope_model,
                )
                visual_off = 0
                if uses_mrope_model:
                    visual_off = 0

                off = 0
                for req_idx, rid in enumerate(req_ids_window):
                    n = int(model_scheduled_list[req_idx])
                    if n <= 0:
                        continue

                    req_state = self.requests.get(rid)
                    global_row_index = int(window_row_indices[req_idx])
                    start_tok = int(self.sequence_buffer.num_computed_tokens[global_row_index])
                    end_tok = start_tok + n

                    if uses_mrope_model and mrope_position_ids_cpu is not None:
                        # mRoPE position ids: use precomputed prompt indices when available, otherwise
                        # fall back to a constant delta-adjusted 1D position broadcast.
                        if (
                            req_state is not None
                            and req_state.prefill_position_ids is not None
                            and start_tok < req_state.num_prompt_tokens
                        ):
                            prompt_end = min(end_tok, req_state.num_prompt_tokens)
                            prompt_n = int(prompt_end - start_tok)
                            if prompt_n > 0:
                                mrope_position_ids_cpu[:, off : off + prompt_n] = req_state.prefill_position_ids[
                                    :, start_tok:prompt_end
                                ]

                            if prompt_n < n:
                                delta = 0
                                if req_state.prefill_rope_deltas is not None:
                                    delta = int(np.asarray(req_state.prefill_rope_deltas).reshape(-1)[0])
                                idxs = np.arange(start_tok + prompt_n, end_tok, dtype=np.int32) + np.int32(delta)
                                mrope_position_ids_cpu[:, off + prompt_n : off + n] = np.broadcast_to(
                                    idxs[None, :], (3, idxs.shape[0])
                                )
                        else:
                            delta = 0
                            if req_state is not None and req_state.prefill_rope_deltas is not None:
                                delta = int(np.asarray(req_state.prefill_rope_deltas).reshape(-1)[0])
                            idxs = np.arange(start_tok, end_tok, dtype=np.int32) + np.int32(delta)
                            mrope_position_ids_cpu[:, off : off + n] = np.broadcast_to(idxs[None, :], (3, n))

                    # Embedding overrides: use precomputed prompt embeddings when available.
                    if (
                        prefill_embeds_cpu is not None
                        and prefill_embeds_mask_cpu is not None
                        and req_state is not None
                        and req_state.prefill_inputs_embeds is not None
                        and start_tok < req_state.num_prompt_tokens
                    ):
                        prompt_end = min(end_tok, req_state.num_prompt_tokens)
                        prompt_n = int(prompt_end - start_tok)
                        if prompt_n > 0:
                            prefill_embeds_cpu[off : off + prompt_n] = req_state.prefill_inputs_embeds[
                                start_tok:prompt_end
                            ]
                            prefill_embeds_mask_cpu[off : off + prompt_n] = True

                            if visual_pos_masks_cpu is not None and req_state.prefill_visual_pos_masks is not None:
                                mask_slice = req_state.prefill_visual_pos_masks[start_tok:prompt_end]
                                visual_pos_masks_cpu[off : off + prompt_n] = mask_slice

                                num_before = int(req_state.prefill_visual_pos_masks[:start_tok].sum())
                                num_in = int(mask_slice.sum())
                                if (
                                    uses_mrope_model
                                    and num_in
                                    and deepstack_visual_embeds_cpu is not None
                                    and req_state.prefill_deepstack_visual_embeds is not None
                                ):
                                    ds_list = req_state.prefill_deepstack_visual_embeds
                                    for layer_idx, buf in enumerate(deepstack_visual_embeds_cpu):
                                        if layer_idx >= len(ds_list):
                                            break
                                        buf[visual_off : visual_off + num_in] = ds_list[layer_idx][
                                            num_before : num_before + num_in
                                        ]
                                    visual_off += num_in

                    off += n

            _pt_group_idx = self._get_full_attention_page_table_index()
            page_table_cpu = self.sequence_buffer.page_table[_pt_group_idx].get_cpu_tensor()
            page_table_version = getattr(self.sequence_buffer.page_table[_pt_group_idx], "cpu_version", None)

            # Preflight check: surface req_id + row details for DP-local page mismatches.
            if dp_size > 1:
                total_pages = int(getattr(self.metadata, "num_pages", 0) or 0)
                page_size = max(1, int(getattr(self.metadata, "page_size", 1)))
                pages_per_shard_opt = pages_per_dp_shard(total_pages, dp_size)
                if pages_per_shard_opt is not None and self.num_reqs_max_model_len % dp_size == 0:
                    rows_per_shard = self.num_reqs_max_model_len // dp_size
                    pages_per_shard = int(pages_per_shard_opt)
                    for local_req_idx in range(num_reqs):
                        req_id_dbg = req_ids_window[local_req_idx]
                        if req_id_dbg is None or int(model_scheduled_list[local_req_idx]) <= 0:
                            continue
                        global_row_index = int(window_row_indices[local_req_idx])
                        seq_len = int(self.sequence_buffer.num_computed_tokens[global_row_index]) + int(
                            model_scheduled_list[local_req_idx]
                        )
                        if seq_len <= 0:
                            continue
                        page_cnt = min((seq_len + page_size - 1) // page_size, int(page_table_cpu.shape[1]))
                        row = np.asarray(page_table_cpu[global_row_index, :page_cnt], dtype=np.int32)
                        row = row[row != 0]
                        if row.size == 0:
                            continue
                        global_req_idx = global_row_index
                        req_shard = min(global_req_idx // rows_per_shard, dp_size - 1)
                        page_lo, page_hi = dp_shard_page_bounds(req_shard, pages_per_shard)
                        invalid = row[(row < page_lo) | (row >= page_hi)]
                        if invalid.size:
                            logger.error(
                                "Pre-execute DP-local mismatch: row=%s req_id=%s req_shard=%s range=[%s, %s) "
                                "sample_bad_page=%s pages_preview=%s scheduled=%s computed=%s",
                                local_req_idx,
                                req_id_dbg,
                                req_shard,
                                page_lo,
                                page_hi,
                                int(invalid[0]),
                                row[:8].tolist(),
                                int(model_scheduled_list[local_req_idx]),
                                int(self.sequence_buffer.num_computed_tokens[global_row_index]),
                            )
                            break

            device_token_handoff: DeviceInputTokenHandoff | None = None
            if pending_pre_async_results is not None and pre_async_repair_pending:
                can_delay_repair = self._can_delay_async_result_repair(
                    scheduler_output=scheduler_output,
                    return_async_output=return_async_output,
                    is_vlm_model=is_vlm_model,
                    frequency_penalties_cpu=self.sequence_buffer.frequency_penalties,
                    presence_penalties_cpu=self.sequence_buffer.presence_penalties,
                    repetition_penalties_cpu=self.sequence_buffer.repetition_penalties,
                )
                if can_delay_repair:
                    device_token_handoff = self._build_device_token_handoff(
                        pre_results=pending_pre_async_results,
                        req_ids_window=req_ids_window,
                        scheduled_list=model_scheduled_list,
                        window_row_indices=window_row_indices,
                        num_tokens_static=num_tokens_static,
                    )
                if device_token_handoff is None:
                    prev_async_start = time.time()
                    self._modify_prev_results(pending_pre_async_results)
                    pre_async_repair_pending = False
                    pending_pre_async_results = None
                    prev_async_time += time.time() - prev_async_start

            (
                token_ids_window_cpu,
                num_computed_tokens_window_cpu,
                temperature_window_cpu,
                top_p_window_cpu,
                top_k_window_cpu,
                min_p_window_cpu,
                page_table_window_cpu,
                frequency_penalties_window_cpu,
                presence_penalties_window_cpu,
                repetition_penalties_window_cpu,
                page_table_window_version,
            ) = self._get_window_state_views(
                start_index=start_index,
                row_count=num_reqs,
                page_table_cpu=page_table_cpu,
                page_table_version=page_table_version,
                row_indices=window_row_indices if packed_window_rows else None,
            )

            total_runner_host_time += time.time() - host_start
            spec_decode_active_window = self.drafter is not None and bool(scheduler_output.scheduled_spec_decode_tokens)
            spec_window_needs_snapshot = bool(
                spec_decode_active_window
                # ``or`` is commutative here; put the O(1) flag first so the
                # O(num_layers) ``cache_replay_required()`` view scan is skipped
                # every step whenever recurrent candidate rows are in use.
                and (self.spec_decode_recurrent_candidates or self.spec.cache_replay_required())
            )
            if sequential_greedy_spec_rows:
                spec_window_needs_snapshot = False
                for _row_pos, _rid in enumerate(req_ids_window):
                    if _rid is None:
                        continue
                    _scheduled_specs = scheduler_output.scheduled_spec_decode_tokens.get(_rid, [])
                    if _scheduled_specs and int(_row_pos) not in sequential_greedy_spec_rows:
                        spec_window_needs_snapshot = self.spec.cache_replay_required()
                        break
            if (
                spec_window_needs_snapshot
                and self.spec_decode_recurrent_candidates
                and not self.spec_decode_recurrent_replay
                and not sequential_greedy_spec_rows
            ):
                # Steady-state candidate fast path: the pre-step recurrent
                # snapshot exists only to feed the replay fallbacks
                # (``replay_prefix_sample`` / ``commit_cache_replay``). Those
                # can only trigger when (a) full hidden rows do NOT cover the
                # window, or (b) a spec request schedules more than one real
                # token (the in-model spec lane is skipped, so the candidate
                # commit cannot cover the accepted prefix). In the steady
                # decode state — every spec request schedules exactly
                # 1 real + k draft tokens and the window bucket returns full
                # hidden — the commit always succeeds and the snapshot is a
                # dead full-pool copy of every recurrent layer, every step.
                # Skip it; any window violating the conditions keeps it.
                _full_hidden_ok = int(num_tokens_static) <= int(
                    getattr(self.executor_manager, "full_hidden_state_max_tokens", 0) or 0
                )
                _snapshot_needed = not _full_hidden_ok
                if not _snapshot_needed:
                    for _row_pos, _rid in enumerate(req_ids_window):
                        if _rid is None:
                            continue
                        _specs = scheduler_output.scheduled_spec_decode_tokens.get(_rid, [])
                        if not _specs or any(int(_t) < 0 for _t in _specs):
                            continue
                        _real = int(model_scheduled_list[_row_pos]) - len(_specs)
                        if _real != 1:
                            _snapshot_needed = True
                            break
                spec_window_needs_snapshot = _snapshot_needed
            pre_step_kv_pages = (
                _snapshot_recurrent_state(self.executor_manager.kv_pages) if spec_window_needs_snapshot else None
            )
            window_token_offsets = np.zeros((len(model_scheduled_list),), dtype=np.int32)
            if self.drafter is not None and model_scheduled_list:
                running_off = 0
                for _i, _n in enumerate(model_scheduled_list):
                    window_token_offsets[_i] = running_off
                    running_off += int(_n)
            spec_recurrent_commit_cpu = None
            applied_pending_commit_req_ids: list[str] = []
            if self.spec_decode_recurrent_candidates:
                spec_recurrent_commit_cpu = self._spec_recurrent_commit_cpu
                spec_recurrent_commit_cpu.fill(0)
                for _row_pos, _rid in enumerate(req_ids_window):
                    if _rid is None:
                        continue
                    _prefix = self._pending_spec_recurrent_commit_by_req.get(str(_rid))
                    if _prefix is None or int(_prefix) <= 0:
                        continue
                    spec_recurrent_commit_cpu[0, int(_row_pos)] = 1
                    spec_recurrent_commit_cpu[1, int(_row_pos)] = int(_prefix)
                    applied_pending_commit_req_ids.append(str(_rid))
            step_start = time.time()
            (
                out_tokens_win,
                _valid_mask_win,
                self.input_ids_buf,
                self.position_ids_buf,
                _hidden_states,
                _logits,
                window_metrics,
            ) = self.executor_manager.execute(
                num_tokens=num_tokens_static,
                scheduled_full_cpu=scheduled_full_cpu,
                req_num_tokens_full_cpu=req_num_tokens_np,
                active_mask_full_cpu=active_mask_full_cpu,
                window_row_indices_cpu=window_row_indices_cpu,
                recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
                input_ids_buf=self.input_ids_buf,
                position_ids_buf=self.position_ids_buf,
                padded_num_reqs=padded_num_reqs,
                token_ids_cpu=token_ids_window_cpu,
                num_computed_tokens_cpu=num_computed_tokens_window_cpu,
                temperature_cpu=temperature_window_cpu,
                top_p_cpu=top_p_window_cpu,
                top_k_cpu=top_k_window_cpu,
                min_p_cpu=min_p_window_cpu,
                frequency_penalties_cpu=frequency_penalties_window_cpu,
                presence_penalties_cpu=presence_penalties_window_cpu,
                repetition_penalties_cpu=repetition_penalties_window_cpu,
                page_table_cpu=page_table_window_cpu,
                page_table_version=page_table_window_version,
                spec_recurrent_commit_cpu=spec_recurrent_commit_cpu,
                mrope_position_ids_cpu=mrope_position_ids_cpu,
                prefill_embeds_cpu=prefill_embeds_cpu,
                prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
                visual_pos_masks_cpu=visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
                device_token_handoff=device_token_handoff,
                wait_for_outputs=not needs_async_output,
            )
            if self.spec_decode_recurrent_candidates:
                self._spec_recurrent_commit_cpu.fill(0)
                for _rid in applied_pending_commit_req_ids:
                    self._pending_spec_recurrent_commit_by_req.pop(str(_rid), None)
            if device_token_handoff is not None and pending_pre_async_results is not None and pre_async_repair_pending:
                # The previous async handle will repair CPU placeholders when
                # the lifecycle loop drains it. Doing that repair here would
                # immediately wait on the previous sampled token and erase the
                # point of the device-side PP token handoff.
                pre_async_repair_pending = False
                pending_pre_async_results = None

            # account for device time (blocking already happened inside execute())
            total_step_time += time.time() - step_start
            num_windows += 1
            total_exec_time += float(window_metrics.get("exec_time", 0.0))
            total_sample_time += float(window_metrics.get("sample_time", 0.0))
            total_prep_time += float(window_metrics.get("prep_time", 0.0))
            total_prep_host_time += float(window_metrics.get("prep_host_time", 0.0))
            total_prep_put_time += float(window_metrics.get("prep_put_time", 0.0))
            total_prep_extra_put_time += float(window_metrics.get("prep_extra_put_time", 0.0))
            total_prep_batch_metadata_time += float(window_metrics.get("prep_batch_metadata_time", 0.0))
            total_prep_handoff_time += float(window_metrics.get("prep_handoff_time", 0.0))
            total_prep_sampler_window_time += float(window_metrics.get("prep_sampler_window_time", 0.0))
            total_prep_ensure_variants_time += float(window_metrics.get("prep_ensure_variants_time", 0.0))
            total_prep_pack_inputs_time += float(window_metrics.get("prep_pack_inputs_time", 0.0))
            total_execute_overhead_time += float(window_metrics.get("execute_overhead_time", 0.0))
            total_model_enqueue_time += float(window_metrics.get("model_enqueue_time", 0.0))
            total_sampler_enqueue_time += float(window_metrics.get("sampler_enqueue_time", 0.0))
            total_greedy_argmax_time += float(window_metrics.get("greedy_argmax_time", 0.0))
            total_greedy_argmax_fastpath += int(window_metrics.get("greedy_argmax_fastpath", 0))
            total_logits_wait_time += float(window_metrics.get("logits_wait_time", 0.0))
            total_exec_enqueue_time += float(window_metrics.get("exec_enqueue_time", 0.0))
            total_exec_wait_time += float(window_metrics.get("exec_wait_time", 0.0))
            total_sampler_wait_time += float(window_metrics.get("sampler_wait_time", 0.0))
            total_pp_stage_dispatch_time += float(window_metrics.get("pp_stage_dispatch_time", 0.0))
            total_pp_queue_wait_time += float(window_metrics.get("pp_queue_wait_time", 0.0))
            total_pp_stage_launches += int(window_metrics.get("pp_stage_launches", 0))
            total_pp_stage_compute_time += float(window_metrics.get("pp_stage_compute_time", 0.0))
            total_pp_stage_max_time = max(
                total_pp_stage_max_time,
                float(window_metrics.get("pp_stage_max_time", 0.0)),
            )
            total_pp_prepare_time += float(window_metrics.get("pp_prepare_time", 0.0))
            total_pp_submit_time += float(window_metrics.get("pp_submit_time", 0.0))
            total_pp_assemble_time += float(window_metrics.get("pp_assemble_time", 0.0))
            total_pp_backbone_time += float(window_metrics.get("pp_backbone_time", 0.0))
            total_pp_combine_time += float(window_metrics.get("pp_combine_time", 0.0))
            total_pp_lm_head_time += float(window_metrics.get("pp_lm_head_time", 0.0))
            total_pp_sampler_enqueue_time += float(window_metrics.get("pp_sampler_enqueue_time", 0.0))
            total_pp_microbatch_scratch_time += float(window_metrics.get("pp_microbatch_scratch_time", 0.0))
            total_pp_microbatch_metadata_time += float(window_metrics.get("pp_microbatch_metadata_time", 0.0))
            total_pp_microbatch_handoff_time += float(window_metrics.get("pp_microbatch_handoff_time", 0.0))
            total_pp_sampler_window_time += float(window_metrics.get("pp_sampler_window_time", 0.0))
            total_pp_ensure_variants_time += float(window_metrics.get("pp_ensure_variants_time", 0.0))
            for stage_idx in range(8):
                key = f"pp_stage_{stage_idx}_time"
                if key in window_metrics:
                    pp_stage_times_by_index[stage_idx] = pp_stage_times_by_index.get(stage_idx, 0.0) + float(
                        window_metrics[key]
                    )
                submit_key = f"pp_stage_{stage_idx}_submit_time"
                if submit_key in window_metrics:
                    pp_stage_submit_times_by_index[stage_idx] = pp_stage_submit_times_by_index.get(
                        stage_idx, 0.0
                    ) + float(window_metrics[submit_key])
                assemble_key = f"pp_stage_{stage_idx}_assemble_time"
                if assemble_key in window_metrics:
                    pp_stage_assemble_times_by_index[stage_idx] = pp_stage_assemble_times_by_index.get(
                        stage_idx, 0.0
                    ) + float(window_metrics[assemble_key])
                execute_key = f"pp_stage_{stage_idx}_execute_time"
                if execute_key in window_metrics:
                    pp_stage_execute_times_by_index[stage_idx] = pp_stage_execute_times_by_index.get(
                        stage_idx, 0.0
                    ) + float(window_metrics[execute_key])
            token_buckets_used.add(int(window_metrics.get("token_bucket", num_tokens_static)))
            req_buckets_used.add(int(window_metrics.get("padded_num_reqs", padded_num_reqs)))

            up_wtime = time.time()
            window_entries: list[tuple[int, str, CachedRequestState | None, int | None, int | None, bool]] = []
            for i, rid in enumerate(req_ids_window):
                if rid is None:
                    continue

                out_idx = len(req_ids_all)
                req_ids_all.append(rid)

                req_state = self.requests.get(rid)
                req_idx = self.sequence_buffer.req_id_to_index.get(rid) if req_state is not None else None
                seq_len: int | None = None
                is_valid = False

                if req_state is not None:
                    seq_len = req_state.num_computed_tokens + scheduler_output.num_scheduled_tokens.get(rid, 0)
                    global_row_index = int(window_row_indices[i])
                    target_len = int(self.sequence_buffer.num_tokens[global_row_index])
                    is_valid = int(scheduled_list[i]) > 0 and seq_len >= target_len

                window_entries.append((i, rid, req_state, req_idx, seq_len, is_valid))

                if scheduler_output.async_scheduling:
                    if is_valid:
                        if req_state is None or req_idx is None or seq_len is None:
                            raise RuntimeError(f"Missing runner state for async request {rid!r}")
                        # The token sampled by this step lands right after the tokens
                        # computed in this step: ``num_computed_tokens + scheduled``
                        # (== ``seq_len``). ``num_tokens_no_spec`` is only maintained
                        # by the async placeholder path itself, so it goes stale one
                        # slot behind whenever the previous step for this request was
                        # finalized synchronously (e.g. the prefill step). Using the
                        # stale counter made the next async finalize overwrite the
                        # previous token and left the real append slot as a 0
                        # placeholder that was then fed back into the model.
                        placeholder_idx = int(seq_len)
                        request_seq_lens.append((out_idx, req_idx, req_state, placeholder_idx))
                    else:
                        discard_sampled_tokens_req_indices.append(out_idx)
                elif return_async_output:
                    sync_finalize_entries.append((req_state, req_idx, seq_len))

            if needs_async_output:
                copy_enqueue_start = time.time()
                row_positions = [row_pos for row_pos, *_rest in window_entries]
                async_windows.append(
                    AsyncWindowResult(
                        req_ids=[rid for _, rid, *_rest in window_entries],
                        row_positions=row_positions,
                        sampled_token_ids=jax.copy_to_host_async(out_tokens_win[:padded_num_reqs]),
                        valid_mask=[is_valid for *_, is_valid in window_entries],
                        token_logprobs=(
                            jax.copy_to_host_async(_logits[:num_reqs])
                            if self.enable_sampler_metrics and _logits is not None
                            else None
                        ),
                    )
                )
                total_async_copy_enqueue_time += time.time() - copy_enqueue_start
                total_post_proc_time += time.time() - up_wtime
                start_index = next_start_index
                continue

            token_materialize_start = time.time()
            tokens_np = np.asarray(out_tokens_win)
            hidden_states_for_spec = _hidden_states
            hidden_np_len = (
                int(getattr(hidden_states_for_spec, "shape", (0,))[0]) if hidden_states_for_spec is not None else 0
            )
            _logits_maybe: typing.Any | None = _logits
            logits_np = np.asarray(_logits_maybe) if self.enable_sampler_metrics and _logits_maybe is not None else None
            total_token_materialize_time += time.time() - token_materialize_start

            spec_commit_scheduled_list = [int(n) for n in scheduled_list]
            spec_window_needs_commit = False
            spec_window_requires_replay = False
            rng_after_verify = self.executor_manager.rng_key if spec_decode_active_window else None

            _run_suffix_sample = partial(
                self.spec.run_suffix_sample,
                num_computed_tokens_window_cpu=num_computed_tokens_window_cpu,
                window_row_indices_cpu=window_row_indices_cpu,
                recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
                padded_num_reqs=padded_num_reqs,
                token_ids_window_cpu=token_ids_window_cpu,
                temperature_window_cpu=temperature_window_cpu,
                top_p_window_cpu=top_p_window_cpu,
                top_k_window_cpu=top_k_window_cpu,
                min_p_window_cpu=min_p_window_cpu,
                frequency_penalties_window_cpu=frequency_penalties_window_cpu,
                presence_penalties_window_cpu=presence_penalties_window_cpu,
                repetition_penalties_window_cpu=repetition_penalties_window_cpu,
                page_table_window_cpu=page_table_window_cpu,
                page_table_window_version=page_table_window_version,
                mrope_position_ids_cpu=mrope_position_ids_cpu,
                prefill_embeds_cpu=prefill_embeds_cpu,
                prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
                visual_pos_masks_cpu=visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
            )

            _replay_prefix_sample = partial(
                self.spec.replay_prefix_sample,
                pre_step_kv_pages=pre_step_kv_pages,
                num_computed_tokens_window_cpu=num_computed_tokens_window_cpu,
                num_tokens_static=num_tokens_static,
                window_row_indices_cpu=window_row_indices_cpu,
                recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
                padded_num_reqs=padded_num_reqs,
                token_ids_window_cpu=token_ids_window_cpu,
                temperature_window_cpu=temperature_window_cpu,
                top_p_window_cpu=top_p_window_cpu,
                top_k_window_cpu=top_k_window_cpu,
                min_p_window_cpu=min_p_window_cpu,
                frequency_penalties_window_cpu=frequency_penalties_window_cpu,
                presence_penalties_window_cpu=presence_penalties_window_cpu,
                repetition_penalties_window_cpu=repetition_penalties_window_cpu,
                page_table_window_cpu=page_table_window_cpu,
                page_table_window_version=page_table_window_version,
                mrope_position_ids_cpu=mrope_position_ids_cpu,
                prefill_embeds_cpu=prefill_embeds_cpu,
                prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
                visual_pos_masks_cpu=visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
                window_token_offsets=window_token_offsets,
                total_scheduled=total_scheduled,
                rng_after_verify=rng_after_verify,
            )

            _replay_prefix_sequential = partial(
                self.spec.replay_prefix_sequential,
                pre_step_kv_pages=pre_step_kv_pages,
                num_computed_tokens_window_cpu=num_computed_tokens_window_cpu,
                window_row_indices_cpu=window_row_indices_cpu,
                recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
                padded_num_reqs=padded_num_reqs,
                token_ids_window_cpu=token_ids_window_cpu,
                temperature_window_cpu=temperature_window_cpu,
                top_p_window_cpu=top_p_window_cpu,
                top_k_window_cpu=top_k_window_cpu,
                min_p_window_cpu=min_p_window_cpu,
                frequency_penalties_window_cpu=frequency_penalties_window_cpu,
                presence_penalties_window_cpu=presence_penalties_window_cpu,
                repetition_penalties_window_cpu=repetition_penalties_window_cpu,
                page_table_window_cpu=page_table_window_cpu,
                page_table_window_version=page_table_window_version,
                mrope_position_ids_cpu=mrope_position_ids_cpu,
                prefill_embeds_cpu=prefill_embeds_cpu,
                prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
                visual_pos_masks_cpu=visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
                rng_after_verify=rng_after_verify,
            )

            # --- Batched greedy verify pre-pass -------------------------------
            # Project + argmax the verify rows of EVERY greedy spec request in
            # this window together: ONE LM-head matmul over the pooled
            # ``[sum(k+1), H]`` hidden rows and ONE device->host argmax transfer,
            # instead of N per-request projections and N blocking int() syncs.
            # Each request's slice is stashed in ``batched_verify_by_row`` and
            # consumed by the emission loop below, which then does only host
            # bookkeeping. Gated exactly like the batched drafter: only greedy,
            # spec-eligible, full-hidden-available rows, and only when >= 2 are
            # verified together (a single request stays bit-identical to the
            # per-request path). ``EASURGE_DISABLE_BATCHED_EMIT=1`` forces the
            # legacy per-request project/argmax (A/B + isolated parity proof).
            batched_verify_by_row: dict[int, _BatchedVerifyRow] = {}
            verify_meta_by_row: dict[int, typing.Any] = {}
            batched_emit_enabled = (
                spec_decode_active_window
                and self.drafter is not None
                and hidden_states_for_spec is not None
                and os.environ.get("EASURGE_DISABLE_BATCHED_EMIT", "0") != "1"
            )
            if batched_emit_enabled:
                _flat_indices: list[int] = []
                _pending_bv: list[tuple[int, int, int, typing.Any]] = []
                for _rp, _rid, _rs, _ridx, _sl, _iv in window_entries:
                    if not _iv or _rs is None or _ridx is None:
                        continue
                    _sched_n = int(scheduled_list[_rp])
                    _spec_toks = scheduler_output.scheduled_spec_decode_tokens.get(_rid, [])
                    if not (
                        bool(_spec_toks)
                        and all(int(_t) >= 0 for _t in _spec_toks)
                        and self.spec.is_spec_request(_rs)
                    ):
                        continue
                    if int(_rp) in sequential_greedy_spec_rows and len(_spec_toks) == 1:
                        continue
                    _real_count = _sched_n - len(_spec_toks)
                    if _real_count <= 0:
                        continue
                    if not self.spec.is_greedy_request(_rs):
                        continue
                    _meta_start = time.time()
                    _meta = self.spec.build_verify_metadata(
                        request_id=_rid,
                        row_pos=int(_rp),
                        req_idx=int(_ridx),
                        start_pos=int(num_computed_tokens_window_cpu[_rp]),
                        real_count=int(_real_count),
                        scheduled_draft_tokens=[int(_t) for _t in _spec_toks],
                        token_ids_window_cpu=token_ids_window_cpu,
                        token_offset=int(window_token_offsets[int(_rp)]),
                    )
                    total_spec_meta_time += time.time() - _meta_start
                    if hidden_np_len < int(_meta.bonus_local_index) + 1:
                        continue
                    _idxs = [*_meta.target_local_indices, int(_meta.bonus_local_index)]
                    _pending_bv.append((int(_rp), len(_flat_indices), len(_idxs), _meta))
                    _flat_indices.extend(int(_x) for _x in _idxs)
                if len(_pending_bv) >= 2:
                    _total_rows = len(_flat_indices)
                    # Pad to a stable ``padded_num_reqs * (k+1)`` bucket so the
                    # on-demand LM-head executable count stays bounded (<= one
                    # per request bucket) instead of one per distinct row total.
                    _pad_target = max(_total_rows, int(padded_num_reqs) * (int(self.num_speculative_tokens) + 1))
                    if _total_rows < _pad_target:
                        _flat_indices.extend([_flat_indices[0]] * (_pad_target - _total_rows))
                    _idx_arr = np.asarray(_flat_indices, dtype=np.int32)
                    _project_start = time.time()
                    _gathered_hidden = hidden_states_for_spec[_idx_arr]
                    _batched_logits = self.spec.project_hidden_rows(_gathered_hidden)
                    total_spec_project_time += time.time() - _project_start
                    _argmax_start = time.time()
                    _batched_argmax = np.asarray(jnp.argmax(_batched_logits, axis=-1)).astype(np.int64)
                    total_spec_argmax_sync_time += time.time() - _argmax_start
                    for _rp, _off, _cnt, _meta in _pending_bv:
                        verify_meta_by_row[_rp] = _meta
                        batched_verify_by_row[_rp] = _BatchedVerifyRow(
                            verify_meta=_meta,
                            offset=int(_off),
                            count=int(_cnt),
                            argmaxes=[int(_x) for _x in _batched_argmax[_off : _off + _cnt].tolist()],
                            gathered_hidden=_gathered_hidden,
                            batched_logits=_batched_logits,
                        )
            # ------------------------------------------------------------------

            for row_pos, rid, req_state, req_idx, seq_len, is_valid in window_entries:
                if not is_valid:
                    sampled_token_ids_all.append([])
                    if spec_token_ids_all is not None:
                        spec_token_ids_all.append([])
                    continue

                scheduled_n = int(scheduled_list[row_pos])
                scheduled_spec_tokens = [int(t) for t in scheduler_output.scheduled_spec_decode_tokens.get(rid, [])]
                has_scheduled_specs = (
                    self.drafter is not None
                    and bool(scheduled_spec_tokens)
                    and all(int(t) >= 0 for t in scheduled_spec_tokens)
                    and self.spec.is_spec_request(req_state)
                )

                if has_scheduled_specs:
                    if req_state is None or req_idx is None:
                        sampled_token_ids_all.append([])
                        if spec_token_ids_all is not None:
                            spec_token_ids_all.append([])
                        continue
                    real_count = scheduled_n - len(scheduled_spec_tokens)
                    if real_count <= 0:
                        sampled_token_ids_all.append([])
                        if spec_token_ids_all is not None:
                            spec_token_ids_all.append([])
                        continue

                    if int(row_pos) in sequential_greedy_spec_rows and len(scheduled_spec_tokens) == 1:
                        draft_token = int(scheduled_spec_tokens[0])
                        target_token = int(tokens_np[row_pos])
                        target_hidden = _window_hidden_row(
                            hidden_states_for_spec,
                            row_pos=int(row_pos),
                            token_offset=int(window_token_offsets[int(row_pos)]),
                            token_index=max(0, int(real_count) - 1),
                            total_window_tokens=int(total_scheduled),
                            padded_reqs=int(padded_num_reqs),
                        )
                        if target_token == draft_token:
                            accepted = 1
                            corrected_token, seed_hidden = _run_suffix_sample(
                                int(row_pos),
                                start_len=int(real_count),
                                suffix_len=1,
                            )
                        else:
                            accepted = 0
                            corrected_token = target_token
                            seed_hidden = target_hidden

                        emitted = ([draft_token] if accepted else []) + [int(corrected_token)]
                        sampled_token_ids_all.append(emitted)
                        req_state.output_token_ids.extend(emitted)

                        emit_start = int(num_computed_tokens_window_cpu[row_pos]) + int(real_count)
                        emit_end = emit_start + len(emitted)
                        if emit_end <= self.max_model_len:
                            self.sequence_buffer.token_ids[req_idx, emit_start:emit_end] = np.asarray(
                                emitted,
                                dtype=np.int32,
                            )
                        known_len = int(num_computed_tokens_window_cpu[row_pos]) + int(real_count) + len(emitted)
                        self.sequence_buffer.num_tokens_no_spec[req_idx] = min(known_len, self.max_model_len)
                        self.sequence_buffer.num_tokens[req_idx] = min(known_len, self.max_model_len)

                        self.spec.record_acceptance(
                            rid,
                            accepted=int(accepted),
                            num_drafts=len(scheduled_spec_tokens),
                        )
                        if self.spec.can_batch_draft(req_state) and step_allows_batch and spec_token_ids_all is not None:
                            # Defer to the post-loop batched draft: collect the seed,
                            # reserve this request's positional slot in the output
                            # list, and write the drafts once for the whole batch.
                            spec_idx = len(spec_token_ids_all)
                            spec_token_ids_all.append([])
                            pending_batched_drafts.append(
                                (
                                    rid,
                                    int(req_idx),
                                    int(emitted[-1]),
                                    max(0, int(known_len) - 2),
                                    seed_hidden,
                                    req_state,
                                    int(known_len),
                                    spec_idx,
                                )
                            )
                        else:
                            draft_timer_start = time.time()
                            next_drafts = self.spec.draft_next(
                                req_id=rid,
                                seed_token=int(emitted[-1]),
                                seed_position=max(0, int(known_len) - 2),
                                seed_hidden=seed_hidden,
                                req_state=req_state,
                                # ``req_idx`` is the request's stable sequence-buffer / KV-pool
                                # slot (window-local ``row_pos`` aliases across windows); use it
                                # as the persistent per-request inline-MTP cache row.
                                row_pos=int(req_idx),
                            )
                            total_spec_draft_time += time.time() - draft_timer_start
                            if next_drafts:
                                draft_start = known_len
                                draft_end = min(draft_start + len(next_drafts), self.max_model_len)
                                self.sequence_buffer.token_ids[req_idx, draft_start:draft_end] = np.asarray(
                                    next_drafts[: draft_end - draft_start],
                                    dtype=np.int32,
                                )
                                self.sequence_buffer.num_tokens[req_idx] = draft_end
                            if spec_token_ids_all is not None:
                                spec_token_ids_all.append(next_drafts)
                        if accepted_spec_tokens_by_req is not None:
                            accepted_spec_tokens_by_req[rid] = int(accepted)
                        if hidden_states_by_req is not None:
                            hidden_states_by_req[rid] = seed_hidden
                        self.spec_decode_num_drafts_accepted += int(accepted)
                        self.spec_decode_num_verify_steps += 1
                        spec_commit_scheduled_list[row_pos] = int(real_count + accepted)
                        spec_window_needs_commit = True
                        committed_candidate = True
                        if self.spec_decode_recurrent_candidates and int(real_count) > 0:
                            commit_timer_start = time.time()
                            committed_candidate = self.spec.queue_or_commit_recurrent_candidate_state(
                                req_id=str(rid),
                                row_pos=int(row_pos),
                                prefix_len=int(real_count + accepted),
                                defer=not bool(getattr(self.drafter, "requires_target_kv_cache", False)),
                            )
                            total_spec_commit_time += time.time() - commit_timer_start
                        if not committed_candidate:
                            if self.spec.recurrent_candidate_count() > 0:
                                spec_window_requires_replay = True
                            elif accepted < len(scheduled_spec_tokens) and self.spec.cache_replay_required():
                                spec_window_requires_replay = True
                        tid = int(emitted[-1])
                        continue

                    verify_meta = verify_meta_by_row.get(int(row_pos))
                    if verify_meta is None:
                        _meta_start = time.time()
                        verify_meta = self.spec.build_verify_metadata(
                            request_id=rid,
                            row_pos=int(row_pos),
                            req_idx=int(req_idx),
                            start_pos=int(num_computed_tokens_window_cpu[row_pos]),
                            real_count=int(real_count),
                            scheduled_draft_tokens=scheduled_spec_tokens,
                            token_ids_window_cpu=token_ids_window_cpu,
                            token_offset=int(window_token_offsets[int(row_pos)]),
                        )
                        total_spec_meta_time += time.time() - _meta_start
                    draft_tokens_for_verify = verify_meta.buffer_draft_tokens
                    accepted = 0
                    projected_target_tokens: list[int] = []
                    hidden_rows: jax.Array | None = None
                    greedy_spec = self.spec.is_greedy_request(req_state)
                    recurrent_state_advanced_by_suffix = False
                    required_hidden_len = int(verify_meta.bonus_local_index) + 1
                    can_project_full_hidden = hidden_np_len >= required_hidden_len
                    _bv = batched_verify_by_row.get(int(row_pos)) if can_project_full_hidden else None
                    if _bv is not None:
                        # Batched greedy fast path: this request's verify-row LM-head
                        # projection and greedy argmax were computed once in the
                        # per-step pre-pass over the whole running batch. Consume the
                        # host argmaxes and the shared gathered hidden; no per-request
                        # project / argmax / device->host sync happens here.
                        for draft_idx, draft_token in enumerate(draft_tokens_for_verify):
                            if int(_bv.argmaxes[draft_idx]) != int(draft_token):
                                break
                            accepted += 1
                        corrected_token = int(_bv.argmaxes[accepted])
                        seed_hidden = _bv.gathered_hidden[_bv.offset + accepted]
                        if self.spec_decode_debug_max_traces > 0:
                            self.spec.record_verify_trace(
                                meta=verify_meta,
                                logits_rows=_bv.batched_logits[_bv.offset : _bv.offset + _bv.count],
                                accepted=accepted,
                                corrected_token=corrected_token,
                                greedy=True,
                                source="full-hidden-batched",
                            )
                    elif can_project_full_hidden:
                        spec_hidden_indices = np.asarray(
                            [*verify_meta.target_local_indices, verify_meta.bonus_local_index],
                            dtype=np.int32,
                        )
                        hidden_rows = hidden_states_for_spec[spec_hidden_indices]
                        project_timer_start = time.time()
                        logits_rows = self.spec.project_hidden_rows(hidden_rows)
                        total_spec_project_time += time.time() - project_timer_start
                        if greedy_spec:
                            _argmax_start = time.time()
                            projected_target_tokens = (
                                np.asarray(jnp.argmax(logits_rows, axis=-1)).astype(np.int64).tolist()
                            )
                            total_spec_argmax_sync_time += time.time() - _argmax_start
                            for draft_idx, draft_token in enumerate(draft_tokens_for_verify):
                                if int(projected_target_tokens[draft_idx]) != int(draft_token):
                                    break
                                accepted += 1
                            corrected_token = int(projected_target_tokens[accepted])
                            if self.spec_decode_recurrent_candidates and self.spec_decode_recurrent_replay:
                                # Exact greedy path (default; required for
                                # correctness): advance the live recurrent state via
                                # a sequential-GDN replay and re-derive the corrected
                                # token + seed hidden from it. The "fast" path
                                # (EASURGE_SPEC_RECURRENT_REPLAY=0) instead reused the
                                # fused verify forward's argmax + the deferred
                                # recurrent candidate-commit; that commit is NOT exact
                                # for greedy, so the live GDN state diverged from
                                # plain greedy decode -> greedy spec output diverged
                                # from the target's greedy stream at the first verify
                                # window and was cross-process non-deterministic
                                # (stale/uninitialized candidate rows). See
                                # tests/inference/esurge/test_spec_decode.py::
                                # test_esurge_runner_spec_decode.
                                corrected_token, replay_hidden = _replay_prefix_sequential(
                                    int(row_pos),
                                    prefix_len=int(real_count + accepted),
                                )
                                hidden_rows = hidden_rows.at[accepted].set(replay_hidden)
                                recurrent_state_advanced_by_suffix = True
                        else:
                            draft_fulls = self.spec.draft_full_log_probs_by_req.pop(str(rid), None)
                            if draft_fulls is None or int(draft_fulls.shape[0]) < len(draft_tokens_for_verify):
                                target_fulls = self.spec.filtered_log_probs(logits_rows, req_state)
                                corrected_token = self.spec.sample_distribution(target_fulls[:1], req_state)
                            else:
                                accepted, corrected_token = self.spec.verify_sampled_window(
                                    target_logits=logits_rows,
                                    draft_log_probs=draft_fulls[: len(draft_tokens_for_verify)],
                                    draft_tokens=draft_tokens_for_verify,
                                    req_state=req_state,
                                )
                        seed_hidden = hidden_rows[accepted]
                        self.spec.record_verify_trace(
                            meta=verify_meta,
                            logits_rows=logits_rows,
                            accepted=accepted,
                            corrected_token=corrected_token,
                            greedy=greedy_spec,
                            source="full-hidden",
                        )
                    else:
                        corrected_token = int(tokens_np[row_pos])
                        seed_hidden = _window_hidden_row(
                            hidden_states_for_spec,
                            row_pos=int(row_pos),
                            token_offset=int(window_token_offsets[int(row_pos)]),
                            token_index=max(0, scheduled_n - 1),
                            total_window_tokens=int(total_scheduled),
                            padded_reqs=int(padded_num_reqs),
                        )
                        trace_logits_rows: list[jax.Array] = []
                        want_trace_logits = (
                            self.spec_decode_debug_max_traces > 0
                            and len(self.spec_decode_debug_traces) < self.spec_decode_debug_max_traces
                        )
                        if greedy_spec:
                            for draft_idx, draft_token in enumerate(draft_tokens_for_verify):
                                target_token, target_hidden = _replay_prefix_sample(row_pos, real_count + draft_idx)
                                corrected_token = int(target_token)
                                seed_hidden = target_hidden
                                if want_trace_logits:
                                    trace_logits_rows.append(
                                        self.spec.project_hidden_rows(target_hidden[None, :])[0]
                                    )
                                if int(target_token) != int(draft_token):
                                    break
                                accepted += 1
                            if accepted == len(draft_tokens_for_verify):
                                corrected_token, seed_hidden = _replay_prefix_sample(row_pos, real_count + accepted)
                        else:
                            draft_fulls = self.spec.draft_full_log_probs_by_req.pop(str(rid), None)
                            if draft_fulls is not None and int(draft_fulls.shape[0]) >= len(draft_tokens_for_verify):
                                for draft_idx, draft_token in enumerate(draft_tokens_for_verify):
                                    _target_token, target_hidden = _replay_prefix_sample(
                                        row_pos,
                                        real_count + draft_idx,
                                    )
                                    target_full = self.spec.filtered_log_probs(
                                        self.spec.project_hidden_rows(target_hidden[None, :]),
                                        req_state,
                                    )
                                    if want_trace_logits:
                                        trace_logits_rows.append(target_full[0])
                                    tok = jnp.asarray([int(draft_token)], dtype=jnp.int32)
                                    draft_full = draft_fulls[draft_idx : draft_idx + 1]
                                    d_lp = jnp.take_along_axis(draft_full, tok[:, None], axis=-1).squeeze(-1)
                                    t_lp = jnp.take_along_axis(target_full, tok[:, None], axis=-1).squeeze(-1)
                                    accept_i = accept_or_reject(d_lp, t_lp, self.spec.rng_split(1)[0])
                                    seed_hidden = target_hidden
                                    if int(np.asarray(accept_i).reshape(-1)[0]) == 1:
                                        accepted += 1
                                        continue
                                    corrected_token = int(
                                        np.asarray(
                                            resample_rejected(target_full, draft_full, self.spec.rng_split(1)[0])
                                        ).reshape(-1)[0]
                                    )
                                    break
                                else:
                                    _target_token, seed_hidden = _replay_prefix_sample(row_pos, real_count + accepted)
                                    bonus_full = self.spec.filtered_log_probs(
                                        self.spec.project_hidden_rows(seed_hidden[None, :]),
                                        req_state,
                                    )
                                    corrected_token = self.spec.sample_distribution(bonus_full, req_state)
                            else:
                                # SP1: the drafter distributions are missing/short, so
                                # every draft must be rejected (accepted stays 0).
                                # Resample the target at the FIRST draft position, whose
                                # context is the real tokens only — mirroring the
                                # full-hidden branch (target_fulls[:1]). The pre-set
                                # fallback `corrected_token = tokens_np[row_pos]` is the
                                # bonus-position sample taken with the drafts as context
                                # and must not be emitted.
                                _target_token, target_hidden = _replay_prefix_sample(row_pos, real_count)
                                target_full = self.spec.filtered_log_probs(
                                    self.spec.project_hidden_rows(target_hidden[None, :]),
                                    req_state,
                                )
                                corrected_token = self.spec.sample_distribution(target_full, req_state)
                                seed_hidden = target_hidden
                                if want_trace_logits:
                                    trace_logits_rows.append(target_full[0])

                        self.spec.record_verify_trace(
                            meta=verify_meta,
                            logits_rows=jnp.stack(trace_logits_rows, axis=0) if trace_logits_rows else None,
                            accepted=accepted,
                            corrected_token=corrected_token,
                            greedy=greedy_spec,
                            source="replay",
                        )

                    _emit_write_start = time.time()
                    emitted = [int(t) for t in draft_tokens_for_verify[:accepted]] + [int(corrected_token)]
                    sampled_token_ids_all.append(emitted)
                    req_state.output_token_ids.extend(emitted)

                    emit_start = int(num_computed_tokens_window_cpu[row_pos]) + real_count
                    emit_end = emit_start + len(emitted)
                    if emit_end <= self.max_model_len:
                        self.sequence_buffer.token_ids[req_idx, emit_start:emit_end] = np.asarray(
                            emitted,
                            dtype=np.int32,
                        )
                    known_len = int(num_computed_tokens_window_cpu[row_pos]) + real_count + len(emitted)
                    self.sequence_buffer.num_tokens_no_spec[req_idx] = min(known_len, self.max_model_len)
                    self.sequence_buffer.num_tokens[req_idx] = min(known_len, self.max_model_len)
                    total_spec_emit_write_time += time.time() - _emit_write_start

                    self.spec.record_acceptance(
                        rid,
                        accepted=int(accepted),
                        num_drafts=len(draft_tokens_for_verify),
                    )
                    if self.spec.can_batch_draft(req_state) and step_allows_batch and spec_token_ids_all is not None:
                        # Defer to the post-loop batched draft (see callsite above).
                        spec_idx = len(spec_token_ids_all)
                        spec_token_ids_all.append([])
                        pending_batched_drafts.append(
                            (
                                rid,
                                int(req_idx),
                                int(emitted[-1]),
                                max(0, int(known_len) - 2),
                                seed_hidden,
                                req_state,
                                int(known_len),
                                spec_idx,
                            )
                        )
                    else:
                        draft_timer_start = time.time()
                        next_drafts = self.spec.draft_next(
                            req_id=rid,
                            seed_token=int(emitted[-1]),
                            seed_position=max(0, int(known_len) - 2),
                            seed_hidden=seed_hidden,
                            req_state=req_state,
                            # Stable KV-pool slot (see the sequential-greedy call above).
                            row_pos=int(req_idx),
                        )
                        total_spec_draft_time += time.time() - draft_timer_start
                        if next_drafts:
                            draft_start = known_len
                            draft_end = min(draft_start + len(next_drafts), self.max_model_len)
                            self.sequence_buffer.token_ids[req_idx, draft_start:draft_end] = np.asarray(
                                next_drafts[: draft_end - draft_start],
                                dtype=np.int32,
                            )
                            self.sequence_buffer.num_tokens[req_idx] = draft_end
                        if spec_token_ids_all is not None:
                            spec_token_ids_all.append(next_drafts)
                    if accepted_spec_tokens_by_req is not None:
                        accepted_spec_tokens_by_req[rid] = int(accepted)
                    if hidden_states_by_req is not None:
                        hidden_states_by_req[rid] = seed_hidden
                    self.spec_decode_num_drafts_accepted += int(accepted)
                    self.spec_decode_num_verify_steps += 1
                    spec_commit_scheduled_list[row_pos] = int(real_count + accepted)
                    spec_window_needs_commit = True
                    committed_candidate = True
                    if (
                        self.spec_decode_recurrent_candidates
                        and int(real_count) > 0
                        and not recurrent_state_advanced_by_suffix
                    ):
                        commit_timer_start = time.time()
                        committed_candidate = self.spec.queue_or_commit_recurrent_candidate_state(
                            req_id=str(rid),
                            row_pos=int(row_pos),
                            prefix_len=int(real_count + accepted),
                            defer=not bool(getattr(self.drafter, "requires_target_kv_cache", False)),
                        )
                        total_spec_commit_time += time.time() - commit_timer_start
                    if not committed_candidate:
                        if self.spec.recurrent_candidate_count() > 0:
                            spec_window_requires_replay = True
                        elif accepted < len(draft_tokens_for_verify) and self.spec.cache_replay_required():
                            spec_window_requires_replay = True
                    tid = int(emitted[-1])
                else:
                    tid = int(tokens_np[row_pos])
                    if req_state is not None and seq_len is not None:
                        if req_idx is not None and 0 <= seq_len < self.max_model_len:
                            self.sequence_buffer.token_ids[req_idx, seq_len] = tid
                        sampled_token_ids_all.append([tid])
                        req_state.output_token_ids.append(tid)
                    else:
                        sampled_token_ids_all.append([tid])

                    if spec_token_ids_all is not None:
                        next_drafts = []
                        if (
                            self.drafter is not None
                            and req_state is not None
                            and req_idx is not None
                            and seq_len is not None
                            and self.spec.is_spec_request(req_state)
                        ):
                            seed_hidden = _window_hidden_row(
                                hidden_states_for_spec,
                                row_pos=int(row_pos),
                                token_offset=int(window_token_offsets[int(row_pos)]),
                                token_index=max(0, scheduled_n - 1),
                                total_window_tokens=int(total_scheduled),
                                padded_reqs=int(padded_num_reqs),
                            )
                            known_len = int(seq_len) + 1
                            prefix_input_ids = None
                            prefix_hidden_states = None
                            prefix_position_ids = None
                            token_offset_i = int(window_token_offsets[int(row_pos)])
                            if (
                                scheduled_n > 1
                                and bool(getattr(self.drafter, "supports_prefix_draft", False))
                                and hidden_np_len >= token_offset_i + int(scheduled_n)
                            ):
                                prefix_start = int(num_computed_tokens_window_cpu[int(row_pos)])
                                row_tokens = np.asarray(token_ids_window_cpu[int(row_pos)])
                                shifted = row_tokens[prefix_start + 1 : prefix_start + int(scheduled_n)]
                                if int(shifted.shape[0]) == int(scheduled_n) - 1:
                                    prefix_input_ids = jnp.asarray(
                                        np.concatenate(
                                            [shifted.astype(np.int32), np.asarray([tid], dtype=np.int32)],
                                            axis=0,
                                        )[None, :],
                                        dtype=jnp.int32,
                                    )
                                    prefix_hidden_states = hidden_states_for_spec[
                                        token_offset_i : token_offset_i + int(scheduled_n)
                                    ][None, :, :]
                                    prefix_position_ids = jnp.arange(
                                        prefix_start,
                                        prefix_start + int(scheduled_n),
                                        dtype=jnp.int32,
                                    )[None, :]
                            draft_timer_start = time.time()
                            next_drafts = self.spec.draft_next(
                                req_id=rid,
                                seed_token=tid,
                                seed_position=max(0, int(known_len) - 2),
                                seed_hidden=seed_hidden,
                                req_state=req_state,
                                prefix_input_ids=prefix_input_ids,
                                prefix_hidden_states=prefix_hidden_states,
                                prefix_position_ids=prefix_position_ids,
                                # Stable KV-pool slot (see the sequential-greedy call above).
                                row_pos=int(req_idx),
                            )
                            total_spec_draft_time += time.time() - draft_timer_start
                            self.sequence_buffer.num_tokens_no_spec[req_idx] = min(known_len, self.max_model_len)
                            self.sequence_buffer.num_tokens[req_idx] = min(known_len, self.max_model_len)
                            if next_drafts:
                                draft_start = known_len
                                draft_end = min(draft_start + len(next_drafts), self.max_model_len)
                                self.sequence_buffer.token_ids[req_idx, draft_start:draft_end] = np.asarray(
                                    next_drafts[: draft_end - draft_start],
                                    dtype=np.int32,
                                )
                                self.sequence_buffer.num_tokens[req_idx] = draft_end
                            if hidden_states_by_req is not None:
                                hidden_states_by_req[rid] = seed_hidden
                            if accepted_spec_tokens_by_req is not None:
                                accepted_spec_tokens_by_req.setdefault(rid, 0)
                        spec_token_ids_all.append(next_drafts)

                if self.enable_sampler_metrics and logits_np is not None and row_pos < logits_np.shape[0]:
                    try:
                        token_logprobs[rid] = logits_np[row_pos]
                    except Exception:
                        pass

            if spec_window_needs_commit and spec_window_requires_replay and pre_step_kv_pages is not None:
                commit_timer_start = time.time()
                self.spec.commit_cache_replay(
                    pre_step_kv_pages=pre_step_kv_pages,
                    commit_scheduled_list=spec_commit_scheduled_list,
                    window_row_indices=window_row_indices_cpu,
                    recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
                    num_tokens_static_original=num_tokens_static,
                    padded_num_reqs=padded_num_reqs,
                    token_ids_window_cpu=token_ids_window_cpu,
                    num_computed_tokens_window_cpu=num_computed_tokens_window_cpu,
                    temperature_window_cpu=temperature_window_cpu,
                    top_p_window_cpu=top_p_window_cpu,
                    top_k_window_cpu=top_k_window_cpu,
                    min_p_window_cpu=min_p_window_cpu,
                    frequency_penalties_window_cpu=frequency_penalties_window_cpu,
                    presence_penalties_window_cpu=presence_penalties_window_cpu,
                    repetition_penalties_window_cpu=repetition_penalties_window_cpu,
                    page_table_window_cpu=page_table_window_cpu,
                    page_table_window_version=page_table_window_version,
                    mrope_position_ids_cpu=mrope_position_ids_cpu,
                    prefill_embeds_cpu=prefill_embeds_cpu,
                    prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
                    visual_pos_masks_cpu=visual_pos_masks_cpu,
                    deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
                )
                if rng_after_verify is not None:
                    self.executor_manager.rng_key = rng_after_verify
                total_spec_commit_time += time.time() - commit_timer_start

            total_post_proc_time += time.time() - up_wtime

            start_index = next_start_index

        # Draft every collected greedy inline-MTP request in ONE batched pass over
        # the pooled MTP cache (``k`` batched forwards for the whole running batch,
        # not ``N * k`` batch-1 forwards). Only reached with >= 2 collected requests
        # (see ``step_allows_batch``); single-request steps drafted in-loop through the
        # exact per-request path above and never populate this list. Runs after the
        # window loop so all per-request fallback drafts (prefix / first window) have
        # already advanced their pool rows; the batched pass leaves those rows
        # untouched. The drafts are scattered back into each request's reserved output
        # slot and sequence buffer exactly as the per-request path would have written
        # them.
        if pending_batched_drafts and spec_token_ids_all is not None:
            draft_timer_start = time.time()
            batched_drafts = self.spec.draft_next_batched(
                [
                    (rid, req_idx, seed_token, seed_position, seed_hidden)
                    for (rid, req_idx, seed_token, seed_position, seed_hidden, _rs, _known_len, _spec_idx) in (
                        pending_batched_drafts
                    )
                ]
            )
            for rid, req_idx, _seed_token, _seed_position, _seed_hidden, _req_state, known_len, spec_idx in (
                pending_batched_drafts
            ):
                next_drafts = batched_drafts.get(rid, [])
                if next_drafts:
                    draft_start = known_len
                    draft_end = min(draft_start + len(next_drafts), self.max_model_len)
                    self.sequence_buffer.token_ids[req_idx, draft_start:draft_end] = np.asarray(
                        next_drafts[: draft_end - draft_start],
                        dtype=np.int32,
                    )
                    self.sequence_buffer.num_tokens[req_idx] = draft_end
                spec_token_ids_all[spec_idx] = next_drafts
            total_spec_draft_time += time.time() - draft_timer_start

        req_id_to_row_index = {
            rid: int(req_idx)
            for rid in req_ids_all
            if (req_idx := self.sequence_buffer.req_id_to_index.get(rid)) is not None
        }
        req_id_to_out_index = {rid: i for i, rid in enumerate(req_ids_all)}

        final_output: ModelRunnerOutput | _AsyncExecutionHandle
        if needs_async_output:
            if scheduler_output.async_scheduling:
                self._update_placeholder(
                    discard_sampled_tokens_req_indices,
                    request_seq_lens,
                )
                async_request_seq_lens = list(request_seq_lens)
                async_pre_results = AsyncPreResults(
                    windows=async_windows,
                    request_seq_lens=async_request_seq_lens,
                )
                self._pre_async_results = async_pre_results
            else:
                async_request_seq_lens = []
                async_pre_results = None

            def _finalize_sync_runner_state(sampled_token_ids: list[list[int]]) -> None:
                """Apply sampled tokens to host buffers for the synchronous overlap path.

                Mirrors what the async finalizer does, but for the case where the
                runner produced sampled tokens synchronously (no
                :class:`AsyncPreResults` to consume on the next step). Walks
                ``sync_finalize_entries`` in lockstep with the per-request
                sampled-id lists and writes the last sampled token id into both
                the device-side ``sequence_buffer.token_ids`` row and the host
                ``output_token_ids`` queue of the corresponding
                :class:`CachedRequestState`.
                """
                for sampled_ids, entry in zip(sampled_token_ids, sync_finalize_entries, strict=False):
                    req_state, req_idx, seq_len = entry
                    if not sampled_ids or req_state is None or seq_len is None:
                        continue
                    tid = int(sampled_ids[-1])
                    if req_idx is not None and 0 <= seq_len < self.max_model_len:
                        self.sequence_buffer.token_ids[req_idx, seq_len] = tid
                    req_state.output_token_ids.append(tid)

            async_output = _AsyncExecutionHandle(
                model_runner_output=ModelRunnerOutput(
                    req_ids=req_ids_all,
                    req_id_to_index=req_id_to_out_index,
                    req_id_to_row_index=req_id_to_row_index,
                    sampled_token_ids=[],
                    spec_token_ids=spec_token_ids_all,
                    logprobs=None,
                    prompt_logprobs_dict={rid: None for rid in req_ids_all},
                    finished_sending=None,
                    finished_recving=None,
                    token_logprobs=None,
                    num_accepted_spec_tokens=accepted_spec_tokens_by_req,
                    hidden_states=hidden_states_by_req,
                ),
                windows=async_windows,
                finalize=(
                    partial(
                        self._finalize_async_scheduler_runner_state,
                        request_seq_lens=async_request_seq_lens,
                        expected_pre_results=async_pre_results,
                    )
                    if scheduler_output.async_scheduling
                    else _finalize_sync_runner_state
                ),
            )
            if return_async_output:
                final_output = async_output
            else:
                token_materialize_start = time.time()
                resolved_output = async_output.get_output()
                total_token_materialize_time += time.time() - token_materialize_start
                final_output = resolved_output
                token_logprobs = resolved_output.token_logprobs or token_logprobs
        else:
            final_output = ModelRunnerOutput(
                req_ids=req_ids_all,
                req_id_to_index=req_id_to_out_index,
                req_id_to_row_index=req_id_to_row_index,
                sampled_token_ids=sampled_token_ids_all,
                spec_token_ids=spec_token_ids_all,
                logprobs=None,
                prompt_logprobs_dict={rid: None for rid in req_ids_all},
                finished_sending=None,
                finished_recving=None,
                token_logprobs=token_logprobs or None,
                num_accepted_spec_tokens=accepted_spec_tokens_by_req,
                hidden_states=hidden_states_by_req,
            )

        metrics_start = time.time()
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.record_runner_metrics(
                execution_time=time.time() - execution_start_time,
                batch_size=len(req_ids_all),
                num_tokens=scheduler_output.total_num_scheduled_tokens,
            )
        metrics_time = time.time() - metrics_start

        total_time = time.time() - execution_start_time
        self._perf_iteration += 1

        total_tokens = int(scheduler_output.total_num_scheduled_tokens)
        agg_tps = total_tokens / total_time if total_time > 0 else 0.0
        num_scheduled_reqs = sum(1 for n in scheduler_output.num_scheduled_tokens.values() if int(n) > 0)
        req_tps = agg_tps / num_scheduled_reqs if num_scheduled_reqs > 0 else 0.0
        self._perf_last_agg_tps = agg_tps
        self._perf_last_req_tps = req_tps
        self._perf_last_total_time = total_time
        self._perf_last_total_tokens = total_tokens
        if self._perf_tps_ema is None:
            self._perf_tps_ema = agg_tps
        else:
            self._perf_tps_ema = self._perf_alpha * agg_tps + (1.0 - self._perf_alpha) * self._perf_tps_ema

        def _fmt_bucket(values: set[int]) -> str:
            """Format a set of compile-bucket sizes as a compact log string.

            Returns ``"?"`` for empty sets, the lone value as-is for singletons,
            and ``"min-max"`` for multi-value sets, used purely for human-readable
            performance log lines so wide windows do not spam every distinct
            bucket count.
            """
            if not values:
                return "?"
            if len(values) == 1:
                return str(next(iter(values)))
            vals = sorted(values)
            return f"{vals[0]}-{vals[-1]}"

        num_new = len(scheduler_output.scheduled_new_reqs)
        num_cached = scheduler_output.scheduled_cached_reqs.num_reqs
        num_finished = len(scheduler_output.finished_req_ids)
        self._perf_history.append(
            RunnerPerfSample(
                iteration=self._perf_iteration,
                total_tokens=total_tokens,
                num_scheduled_reqs=num_scheduled_reqs,
                num_new=num_new,
                num_cached=num_cached,
                num_finished=num_finished,
                total_time=total_time,
                agg_tps=agg_tps,
                req_tps=req_tps,
                ema_tps=float(self._perf_tps_ema),
            )
        )

        step_gap_time = total_step_time - (total_prep_time + total_exec_time + total_sample_time)
        step_gap_time = max(0.0, step_gap_time)

        misc_time = total_time - (
            updating_states_time
            + prev_async_time
            + total_runner_host_time
            + total_async_copy_enqueue_time
            + total_token_materialize_time
            + total_post_proc_time
            + total_prep_time
            + total_exec_time
            + total_sample_time
            + total_execute_overhead_time
            + step_gap_time
            + metrics_time
        )
        misc_time = max(0.0, misc_time)

        self._perf_phase_history.append(
            {
                "iteration": self._perf_iteration,
                "total_tokens": total_tokens,
                "num_scheduled_reqs": num_scheduled_reqs,
                "num_new": num_new,
                "num_cached": num_cached,
                "num_finished": num_finished,
                "num_windows": num_windows,
                "token_buckets": sorted(int(v) for v in token_buckets_used),
                "req_buckets": sorted(int(v) for v in req_buckets_used),
                "agg_tps": agg_tps,
                "req_tps": req_tps,
                "ema_tps": float(self._perf_tps_ema),
                "total_time": total_time,
                "runner_time": total_runner_host_time,
                "copy_enqueue_time": total_async_copy_enqueue_time,
                "token_materialize_time": total_token_materialize_time,
                "prep_time": total_prep_time,
                "prep_host_time": total_prep_host_time,
                "prep_put_time": total_prep_put_time,
                "prep_extra_put_time": total_prep_extra_put_time,
                "prep_batch_metadata_time": total_prep_batch_metadata_time,
                "prep_handoff_time": total_prep_handoff_time,
                "prep_sampler_window_time": total_prep_sampler_window_time,
                "prep_ensure_variants_time": total_prep_ensure_variants_time,
                "prep_pack_inputs_time": total_prep_pack_inputs_time,
                "forward_time": total_exec_time,
                "model_enqueue_time": total_model_enqueue_time,
                "sampler_enqueue_time": total_sampler_enqueue_time,
                "greedy_argmax_time": total_greedy_argmax_time,
                "greedy_argmax_fastpath": total_greedy_argmax_fastpath,
                "logits_wait_time": total_logits_wait_time,
                "exec_enqueue_time": total_exec_enqueue_time,
                "exec_wait_time": total_exec_wait_time,
                "sample_time": total_sample_time,
                "sampler_wait_time": total_sampler_wait_time,
                "execute_overhead_time": total_execute_overhead_time,
                "metrics_time": metrics_time,
                "prev_async_time": prev_async_time,
                "step_time": total_step_time,
                "step_gap_time": step_gap_time,
                "sync_time": updating_states_time,
                "post_time": total_post_proc_time,
                "spec_project_time": total_spec_project_time,
                "spec_argmax_sync_time": total_spec_argmax_sync_time,
                "spec_meta_time": total_spec_meta_time,
                "spec_emit_write_time": total_spec_emit_write_time,
                "spec_draft_time": total_spec_draft_time,
                "spec_suffix_time": self._spec_suffix_time_acc,
                "spec_replay_time": self._spec_replay_time_acc,
                "spec_commit_time": total_spec_commit_time,
                "misc_time": misc_time,
                "pp_stage_dispatch_time": total_pp_stage_dispatch_time,
                "pp_queue_wait_time": total_pp_queue_wait_time,
                "pp_stage_launches": total_pp_stage_launches,
                "pp_stage_compute_time": total_pp_stage_compute_time,
                "pp_stage_max_time": total_pp_stage_max_time,
                "pp_prepare_time": total_pp_prepare_time,
                "pp_submit_time": total_pp_submit_time,
                "pp_assemble_time": total_pp_assemble_time,
                "pp_backbone_time": total_pp_backbone_time,
                "pp_combine_time": total_pp_combine_time,
                "pp_lm_head_time": total_pp_lm_head_time,
                "pp_sampler_enqueue_time": total_pp_sampler_enqueue_time,
                "pp_microbatch_scratch_time": total_pp_microbatch_scratch_time,
                "pp_microbatch_metadata_time": total_pp_microbatch_metadata_time,
                "pp_microbatch_handoff_time": total_pp_microbatch_handoff_time,
                "pp_sampler_window_time": total_pp_sampler_window_time,
                "pp_ensure_variants_time": total_pp_ensure_variants_time,
            }
        )

        prep_detail = ""
        if (total_prep_host_time + total_prep_put_time + total_prep_extra_put_time) > 0:
            prep_detail = (
                f"(host={total_prep_host_time * 1e3:.2f}ms put={total_prep_put_time * 1e3:.2f}ms "
                f"extra={total_prep_extra_put_time * 1e3:.2f}ms) "
            )
        if (
            total_prep_batch_metadata_time
            + total_prep_handoff_time
            + total_prep_sampler_window_time
            + total_prep_ensure_variants_time
            + total_prep_pack_inputs_time
        ) > 0:
            prep_detail += (
                f"(batch={total_prep_batch_metadata_time * 1e3:.2f}ms "
                f"handoff={total_prep_handoff_time * 1e3:.2f}ms "
                f"samplerwin={total_prep_sampler_window_time * 1e3:.2f}ms "
                f"ensure={total_prep_ensure_variants_time * 1e3:.2f}ms "
                f"pack={total_prep_pack_inputs_time * 1e3:.2f}ms) "
            )

        queue_detail = (
            f"q(run={int(getattr(scheduler_output, 'num_running_reqs', 0))},"
            f"wait={int(getattr(scheduler_output, 'num_waiting_reqs', 0))},"
            f"freep={getattr(scheduler_output, 'free_pages', '?')},"
            f"budget={getattr(scheduler_output, 'token_budget_remaining', '?')}/"
            f"{getattr(scheduler_output, 'token_budget_initial', '?')}) "
        )
        pp_detail = (
            f"pp(stage={total_pp_stage_launches},dispatch={total_pp_stage_dispatch_time * 1e3:.2f}ms,"
            f"queue={total_pp_queue_wait_time * 1e3:.2f}ms)"
        )
        if (
            total_pp_prepare_time
            + total_pp_submit_time
            + total_pp_assemble_time
            + total_pp_backbone_time
            + total_pp_combine_time
            + total_pp_lm_head_time
            + total_pp_sampler_enqueue_time
        ) > 0:
            pp_detail += (
                f" ppd(prep={total_pp_prepare_time * 1e3:.1f}ms,submit={total_pp_submit_time * 1e3:.1f}ms,"
                f"asm={total_pp_assemble_time * 1e3:.1f}ms,bb={total_pp_backbone_time * 1e3:.1f}ms,"
                f"comb={total_pp_combine_time * 1e3:.1f}ms,lm={total_pp_lm_head_time * 1e3:.1f}ms,"
                f"sampq={total_pp_sampler_enqueue_time * 1e3:.1f}ms)"
            )
        if pp_stage_submit_times_by_index:
            submit_detail = ",".join(
                f"s{stage_idx}={stage_time * 1e3:.1f}ms"
                for stage_idx, stage_time in sorted(pp_stage_submit_times_by_index.items())
            )
            pp_detail += f" pp_submit({submit_detail})"
        if pp_stage_assemble_times_by_index or pp_stage_execute_times_by_index:
            assemble_detail = ",".join(
                f"s{stage_idx}={stage_time * 1e3:.1f}ms"
                for stage_idx, stage_time in sorted(pp_stage_assemble_times_by_index.items())
            )
            execute_detail = ",".join(
                f"s{stage_idx}={stage_time * 1e3:.1f}ms"
                for stage_idx, stage_time in sorted(pp_stage_execute_times_by_index.items())
            )
            pp_detail += f" pp_split(asm=[{assemble_detail}],jit=[{execute_detail}])"
        if (
            total_pp_microbatch_scratch_time
            + total_pp_microbatch_metadata_time
            + total_pp_microbatch_handoff_time
            + total_pp_sampler_window_time
            + total_pp_ensure_variants_time
        ) > 0:
            pp_detail += (
                f" ppmb(scratch={total_pp_microbatch_scratch_time * 1e3:.1f}ms,"
                f"meta={total_pp_microbatch_metadata_time * 1e3:.1f}ms,"
                f"handoff={total_pp_microbatch_handoff_time * 1e3:.1f}ms,"
                f"samplerwin={total_pp_sampler_window_time * 1e3:.1f}ms,"
                f"ensure={total_pp_ensure_variants_time * 1e3:.1f}ms)"
            )
        if total_pp_stage_compute_time > 0:
            stage_detail = ",".join(
                f"s{stage_idx}={stage_time * 1e3:.1f}ms"
                for stage_idx, stage_time in sorted(pp_stage_times_by_index.items())
            )
            pp_detail = (
                f"pp(stage={total_pp_stage_launches},compute={total_pp_stage_compute_time * 1e3:.2f}ms,"
                f"max={total_pp_stage_max_time * 1e3:.2f}ms" + (f",{stage_detail}" if stage_detail else "") + ")"
            )

        self.log_it(
            f"[perf] it={self._perf_iteration:06d} "
            f"win={num_windows} "
            f"reqs={len(req_ids_all)}(new={num_new},cached={num_cached},fin={num_finished},pad={_fmt_bucket(req_buckets_used)}) "
            f"tok={total_tokens}/b{_fmt_bucket(token_buckets_used)} "
            f"{queue_detail}"
            f"agg_tps={agg_tps:,.0f} req_tps={req_tps:,.1f} ema={self._perf_tps_ema:,.0f} "
            f"{pp_detail} "
            f"runner={total_runner_host_time * 1e3:.2f}ms "
            f"copyq={total_async_copy_enqueue_time * 1e3:.2f}ms "
            f"token_wait={total_token_materialize_time * 1e3:.2f}ms "
            f"prep={total_prep_time * 1e3:.2f}ms {prep_detail}"
            f"fwd={total_exec_time * 1e3:.2f}ms samp={total_sample_time * 1e3:.2f}ms "
            f"ovh={total_execute_overhead_time * 1e3:.2f}ms metrics={metrics_time * 1e3:.2f}ms "
            f"async={prev_async_time * 1e3:.2f}ms "
            f"step={total_step_time * 1e3:.2f}ms gap={step_gap_time * 1e3:.2f}ms "
            f"sync={updating_states_time * 1e3:.2f}ms post={total_post_proc_time * 1e3:.2f}ms misc={misc_time * 1e3:.2f}ms "
            f"total={total_time * 1e3:.2f}ms"
        )

        return final_output

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute the model synchronously on scheduled requests.

        Main public entry point for model execution. Delegates to the internal
        implementation which handles state synchronization, batch processing,
        and token generation.

        Args:
            scheduler_output: Output from the scheduler containing requests to
                process, tokens to generate per request, and lifecycle information.

        Returns:
            ModelRunnerOutput containing request IDs, sampled tokens, and optional
            log probabilities and timing information.

        Note:
            This is the synchronous version. For async execution that allows
            overlapping with scheduling, use execute_model_async() instead.
        """
        return typing.cast(ModelRunnerOutput, self._execute_model_impl(scheduler_output))

    def execute_model_async(self, scheduler_output: SchedulerOutput) -> _AsyncExecutionHandle:
        """Dispatch model work and defer the host-side token materialization.

        TPU/JAX dispatch is already asynchronous on the calling thread. This
        method exploits that by keeping execution on the scheduler thread,
        returning an async handle once the device work and host copies have been
        queued, and letting the lifecycle loop do scheduler prefetch work before
        calling wait_for_execution().

        Args:
            scheduler_output: Scheduler decision for the current step.

        Returns:
            An :class:`_AsyncExecutionHandle` whose ``get_output()`` blocks on
            the device-to-host copies and finalizes the
            :class:`ModelRunnerOutput`.
        """
        return typing.cast(_AsyncExecutionHandle, self._execute_model_impl(scheduler_output, return_async_output=True))

    def initialize_async_executor(self) -> None:
        """Retire any legacy background executor and confirm same-thread overlap.

        Historical context: an earlier version of the overlap path moved
        compiled-step dispatch to a :class:`ThreadPoolExecutor` so the
        scheduler thread could prepare the next batch while the current
        one was in flight. TPU/JAX dispatch turned out to be unreliable
        across threads (different XLA-runtime contexts), so the design
        was changed to keep dispatch on the scheduler thread and use
        host-async copies (:meth:`execute_model_async`) for the overlap.
        This method is retained for API compatibility: if a stale
        executor is still attached (older callers), it is shut down
        cleanly; otherwise it is a no-op that just records the chosen
        overlap strategy.
        """
        if self._executor is not None:
            logger.debug("Shutting down legacy async executor")
            self._executor.shutdown(wait=True)
            self._executor = None
        logger.debug("Using same-thread async execution handles for overlap")

    def reset_state(self) -> None:
        """Forget every in-flight request and drop pending async results.

        Wipes the runner's three state containers:

        * ``self.requests`` — the dict of :class:`CachedRequestState` that
          tracks per-request prompt tokens, vision data, and sample state.
        * ``self.sequence_buffer`` — the per-row scheduler view (token ids,
          positions, sampling params) backing the model step.
        * ``self._pre_async_results`` — any deferred sampled-token payload
          from the previous overlap window.

        Required precondition (callers should enforce): no requests are
        currently in flight, otherwise their device-side rows will become
        unreachable. The method itself does not check, so the engine
        :meth:`update_model_weights` / :meth:`pause` paths gate the call
        on ``num_running_requests + num_pending_requests == 0``.
        """
        self.requests.clear()
        self.sequence_buffer.clear()
        self._pre_async_results = None
        self.spec.on_reset()

    def wait_for_execution(self, future: Future | _AsyncExecutionHandle) -> ModelRunnerOutput:
        """Wait for an async execution to complete and return the result.

        Args:
            future: The async handle returned by execute_model_async()

        Returns:
            ModelRunnerOutput: The completed model execution output

        Note:
            This call blocks until sampled tokens have been copied to the host
            and any deferred runner-side state updates have been applied.
        """
        if isinstance(future, _AsyncExecutionHandle):
            return future.get_output()
        return future.result()

    def shutdown(self) -> None:
        """Tear down legacy thread executor and forward to executor manager.

        Engine-teardown path: drains any leftover background executor
        retained for backward compat, then defers to
        :meth:`ExecutionManager.shutdown` so resident PP-stage worker
        threads inside :class:`ModelStepExecutor` are joined. Safe to
        invoke when the runner has already been shut down.
        """
        if self._executor is not None:
            logger.debug("Shutting down async executor")
            self._executor.shutdown(wait=True)
            self._executor = None
        if getattr(self, "executor_manager", None) is not None:
            self.executor_manager.shutdown()
