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

import time
import typing
from bisect import bisect_left
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass

import jax
import numpy as np
import spectrax as spx
from eformer.loggings import get_logger
from jax import numpy as jnp
from jax.experimental import multihost_utils

from easydel.inference.esurge.config import KernelTilePolicy, PipelineInferenceMode
from easydel.infra.sharding import replicated_named_sharding
from easydel.layers.quantization import TurboQuantConfig
from easydel.operations.kernels.inference_gdn import set_gdn_kernel_tile_policy

from ..core.dp_sharding import dp_shard_for_page_id, dp_shard_page_bounds, pages_per_dp_shard
from ..core.interface import create_kv_cache_specs_from_config, estimate_runtime_page_budget
from ..metrics import get_metrics_collector
from ..outputs import ModelRunnerOutput
from ..scheduler import SchedulerOutput
from ..utils import model_uses_mrope
from .async_types import AsyncPreResults, AsyncWindowResult
from .execution_manager import ExecutionManager
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
from .states import CachedRequestState

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule
    from easydel.infra.etils import MpMdSchedulers

logger = get_logger("eSurge")


@dataclass(frozen=True)
class RunnerPerfSample:
    iteration: int
    total_tokens: int
    num_scheduled_reqs: int
    num_new: int
    num_cached: int
    num_finished: int
    total_time: float
    agg_tps: float
    req_tps: float
    ema_tps: float


class _AsyncExecutionHandle:
    """Deferred host-materialized model output for overlap execution."""

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        windows: list[AsyncWindowResult],
        finalize: typing.Callable[[list[list[int]]], None] | None = None,
    ) -> None:
        self._model_runner_output = model_runner_output
        self._windows = windows
        self._finalize = finalize
        self._resolved_output: ModelRunnerOutput | None = None

    def get_output(self) -> ModelRunnerOutput:
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


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int, min_input_pad: int) -> int:  # pyright: ignore[reportUnusedFunction]
    """Calculate padded request count for compilation efficiency.

    Pads the number of requests to the nearest power of 2 that is at least
    ``min_input_pad``.  This reduces the number of unique compilations
    needed while maintaining good utilization.

    Args:
        x: Actual number of requests.
        upper_limit: Maximum allowed requests.
        min_input_pad: Minimum padding floor; values of ``x`` at or below
            this threshold are padded up to ``min_input_pad``.

    Returns:
        Padded request count, capped at ``upper_limit``.

    Example:
        >>> _get_padded_num_reqs_with_upper_limit(3, 32, 8)   # Returns 8
        >>> _get_padded_num_reqs_with_upper_limit(10, 32, 8)  # Returns 16
        >>> _get_padded_num_reqs_with_upper_limit(20, 16, 8)  # Returns 16
    """
    res = min_input_pad if x <= min_input_pad else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


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
        pipeline_inference: PipelineInferenceMode = "auto",
        max_cache_tokens: int | None = None,
        cache_capacity_margin: float = 0.92,
        kernel_tile_policy: KernelTilePolicy = "auto",
        max_model_len: int = 2**13,
        max_num_batched_tokens: int | None = None,
        min_input_pad: int = 256,
        min_token_pad: int | None = None,
        max_num_seqs: int = 16,
        max_num_seq_buckets: list[int] | None = None,
        async_scheduling: bool = True,
        use_aot_forward: bool = True,
        bind_graphstate_for_aot: bool = False,
        verbose: bool = False,
        enable_overlap_execution: bool = True,
        enable_sampler_metrics: bool = False,
        enable_window_aware_runtime_cap: bool = False,
        mpmd_scheduler: MpMdSchedulers | None = None,
    ):
        """Initialize the model runner.

        Args:
            model: EasyDeL model instance.
            hbm_utilization: Target cache-memory utilization ratio.
            page_size: KV page size used by cache metadata.
            max_model_len: Maximum sequence length.
            max_num_batched_tokens: Maximum scheduler token budget used for
                window-aware runtime-cap estimation.
            min_input_pad: Minimum request-count bucket.
            min_token_pad: Optional minimum token bucket size.
            max_num_seqs: Maximum concurrent sequences.
            max_num_seq_buckets: Optional explicit request-count buckets.
            async_scheduling: Whether scheduler async token sampling is enabled.
            use_aot_forward: Whether to use AOT compilation.
            bind_graphstate_for_aot: Whether model-step AOT executables
                capture graphstate/graphother as compile-time constants.
                Default: False.
            verbose: Enable verbose execution logs.
            enable_overlap_execution: Enable overlap execution path.
            enable_sampler_metrics: Enable sampler-side metrics.
            enable_window_aware_runtime_cap: Whether to derive the runtime
                request cap from the model's live KV-window page demand.
                When False, the runner falls back to the cache metadata's
                heuristic request cap instead.
        """
        self.model = model.esurge_compatible_model
        logger.debug(f"Initializing eSurgeRunner with {max_model_len=}, {max_num_seqs=}")
        logger.debug(f"Configuration: {hbm_utilization=}, {page_size=}")
        self.pipeline_plan = build_pipeline_inference_plan(
            model=self.model,
            mpmd_scheduler=mpmd_scheduler,
            pipeline_inference=pipeline_inference,
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
            )
        else:
            self.metadata = self.model.create_ragged_page_cache_config(
                hbm_utilization=hbm_utilization,
                page_size=page_size,
                max_length=max_model_len,
            )
        cap_metadata_pages(self.metadata, self.pipeline_plan)
        self.max_num_batched_tokens = (
            int(max_model_len)
            if max_num_batched_tokens is None
            else max(1, min(int(max_num_batched_tokens), int(max_model_len)))
        )
        self.enable_window_aware_runtime_cap = bool(enable_window_aware_runtime_cap)
        self.max_model_len = max_model_len
        self.kv_cache_groups = self._build_kv_cache_groups()
        self.window_aware_runtime_estimate = self._apply_window_aware_runtime_cap(self.max_num_batched_tokens)
        self.max_num_seq_buckets = self._init_seq_buckets(max_num_seq_buckets, max_num_seqs, min_input_pad)
        self.max_num_seqs = max_num_seqs
        self.max_num_reqs = self.max_num_seq_buckets[-1]
        self.async_scheduling = bool(async_scheduling)
        self.min_input_pad = max(min_input_pad, self.max_num_seq_buckets[0])
        self.page_size = int(self.metadata.page_size)
        self.max_pages_per_req = int(self.metadata.max_num_pages_per_req)

        if min_token_pad is None:
            # Request-count padding and token-count padding have different
            # latency tradeoffs.  In PP inference a decode step is commonly
            # one token; tying token buckets to min_input_pad forces those
            # steps through b4/b8/... backbone executables and directly
            # increases per-token latency.
            min_token_pad_i = 1 if self.pipeline_plan.is_enabled else self.min_input_pad
        else:
            min_token_pad_i = int(min_token_pad)
        min_token_pad_i = min(min_token_pad_i, int(self.max_model_len))
        self.num_tokens_paddings = self._get_token_paddings(
            min_token_size=min_token_pad_i,
            max_token_size=self.max_model_len,
            padding_gap=0,
        )
        self.max_num_tokens = self.num_tokens_paddings[-1]

        logger.debug("Creating ExecutionManager and initializing pages cache")
        manager_cls = PipelineExecutionManager if self.pipeline_plan.is_enabled else ExecutionManager
        self.executor_manager = manager_cls(
            model=self.model,
            use_aot_forward=use_aot_forward,
            bind_graphstate_for_aot=bind_graphstate_for_aot,
            min_input_pad=self.min_input_pad,
            max_model_len=max_model_len,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            metadata=self.metadata,
            verbose=verbose,
            mpmd_scheduler=mpmd_scheduler,
            pipeline_plan=self.pipeline_plan,
        )
        self.log_it = logger.info if verbose else logger.debug
        self._setup_variables()
        self.enable_overlap_execution = enable_overlap_execution
        self.enable_sampler_metrics = enable_sampler_metrics

        # Perf logging state (kept lightweight; no allocations in the hot path).
        self._perf_iteration = 0
        self._perf_tps_ema: float | None = None
        self._perf_alpha = 0.2
        self._perf_last_agg_tps: float | None = None
        self._perf_last_req_tps: float | None = None
        self._perf_last_total_time: float | None = None
        self._perf_last_total_tokens: int | None = None
        self._perf_history: deque[RunnerPerfSample] = deque(maxlen=max(32768, int(max_model_len) * 4))

        # Async scheduling state
        self._pre_async_results: AsyncPreResults | None = None
        self._executor: typing.Any = None  # ThreadPoolExecutor, typed as Any to avoid circular import
        logger.debug("eSurgeRunner initialization complete")
        self._log_startup_summary()

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
        """Reset runtime-cap metadata to the default non-window-aware state."""
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

    @property
    def mesh(self):
        """Get the JAX sharding mesh from the model.

        Returns:
            The JAX mesh used for distributed execution.
        """
        return self.model.mesh

    @property
    def _empty_sharding(self):
        """Get empty sharding for replicated arrays.

        Returns:
            NamedSharding with empty PartitionSpec for fully replicated arrays.
        """
        return replicated_named_sharding(self.mesh)

    @staticmethod
    def _get_token_paddings(min_token_size: int, max_token_size: int, padding_gap: int) -> list[int]:
        """Generate padding sizes for efficient compilation.

        Args:
            min_token_size: Minimum token size (must be power of 2)
            max_token_size: Maximum token size to cover
            padding_gap: Gap between padding sizes (0 for exponential growth)

        Returns:
            List of padding sizes
        """
        if not ((min_token_size & (min_token_size - 1) == 0) and min_token_size > 0):
            logger.error(f"Invalid min_token_size={min_token_size}, must be power of 2")
            raise ValueError(f"min_token_size must be a power of 2, got {min_token_size}")
        paddings = []
        num = min_token_size

        if padding_gap == 0:
            while num <= max_token_size:
                paddings.append(num)
                num *= 2
        else:
            while num <= padding_gap:
                paddings.append(num)
                num *= 2
            num //= 2
            while num < max_token_size:
                num += padding_gap
                paddings.append(num)
        if paddings[-1] != max_token_size:
            paddings.append(max_token_size)
        return paddings

    @staticmethod
    def _get_request_paddings(min_bucket: int, max_bucket: int) -> list[int]:
        """Generate request count buckets using exponential growth.

        Args:
            min_bucket: Minimum bucket size.
            max_bucket: Maximum bucket size (must be included).

        Returns:
            List of bucket sizes from min_bucket to max_bucket,
            doubling at each step.
        """
        min_bucket = max(1, min(min_bucket, max_bucket))
        buckets: list[int] = []
        current = min_bucket
        while current < max_bucket:
            buckets.append(current)
            current *= 2
        if not buckets or buckets[-1] != max_bucket:
            buckets.append(max_bucket)
        return buckets

    def _init_seq_buckets(
        self,
        user_buckets: list[int] | None,
        max_num_seqs: int,
        min_input_pad: int,
    ) -> list[int]:
        """Initialize sequence count buckets for compilation.

        Args:
            user_buckets: Optional user-provided bucket sizes.
            max_num_seqs: Maximum number of sequences.
            min_input_pad: Minimum input padding.

        Returns:
            Sorted list of bucket sizes, always including max_num_seqs.
        """
        if user_buckets:
            buckets = sorted({int(b) for b in user_buckets if 0 < int(b) <= max_num_seqs})
        else:
            buckets = self._get_request_paddings(min_input_pad, max_num_seqs)
        if not buckets or buckets[-1] != max_num_seqs:
            buckets.append(max_num_seqs)
        return buckets

    def _get_current_bucket(self, num_reqs: int) -> int:
        """Select the smallest bucket that can accommodate num_reqs.

        Args:
            num_reqs: Number of active requests

        Returns:
            Smallest sufficient bucket size from the active runtime buckets.
        """
        buckets = getattr(self, "active_num_seq_buckets", self.max_num_seq_buckets)
        if num_reqs <= 0:
            return buckets[0]
        for bucket in buckets:
            if num_reqs <= bucket:
                return bucket
        return buckets[-1]

    @staticmethod
    def _clamp_request_buckets_to_runtime_cap(buckets: list[int], runtime_cap: int) -> list[int]:
        """Clamp request-count buckets to the runtime execution cap.

        The runner may admit more requests globally than it can execute in a
        single scheduler window.  Compilation and bucket lookup should
        therefore only consider request-count buckets that are reachable
        under the current runtime window cap.

        Args:
            buckets: Original list of request-count bucket sizes.
            runtime_cap: Maximum number of requests executable in one
                scheduler window.

        Returns:
            Sorted list of bucket sizes where every entry is at most
            ``runtime_cap``, with ``runtime_cap`` itself always included
            as the final element.
        """
        runtime_cap = max(1, int(runtime_cap))
        clamped = sorted({int(bucket) for bucket in buckets if 0 < int(bucket) <= runtime_cap})
        if not clamped or clamped[-1] != runtime_cap:
            clamped.append(runtime_cap)
        return clamped

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
        self.executor_manager.invalidate_sampler_penalty_state(
            self.sequence_buffer.token_ids,
            self.sequence_buffer.num_tokens,
        )

        # VLM host-side scratch buffers keyed by `num_tokens_static` (avoid repeated
        # large allocations while keeping the step-function input pytree stable).
        self._vlm_cpu_buffers: dict[
            int,
            tuple[
                np.ndarray | None,  # prefill_embeds_cpu
                np.ndarray | None,  # prefill_embeds_mask_cpu
                np.ndarray | None,  # mrope_position_ids_cpu
                np.ndarray | None,  # visual_pos_masks_cpu
                list[np.ndarray] | None,  # deepstack_visual_embeds_cpu
            ],
        ] = {}

    def _get_vlm_cpu_buffers(
        self,
        *,
        num_tokens_static: int,
        uses_mrope_model: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[np.ndarray] | None]:
        """Get or create cached CPU buffers for VLM prefill data.

        Retrieves pre-allocated CPU buffers for VLM embedding overrides,
        keyed by num_tokens_static to avoid repeated allocations while
        keeping the step function input shape stable.

        Args:
            num_tokens_static: Token count bucket size for buffer sizing.
            uses_mrope_model: Whether the model uses mRoPE positions.

        Returns:
            Tuple of (prefill_embeds_cpu, prefill_embeds_mask_cpu,
            mrope_position_ids_cpu, visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu) where optional arrays are
            None if not applicable.

        Side Effects:
            - Creates and caches buffers in self._vlm_cpu_buffers.
            - Clears masks/position buffers each call (filled by caller).
        """
        num_tokens_static = int(num_tokens_static)
        cached = self._vlm_cpu_buffers.get(num_tokens_static)
        if cached is None:
            hidden_size = int(getattr(self.model.config.get_text_config(), "hidden_size", 0) or 1)

            prefill_embeds_cpu = np.zeros((num_tokens_static, hidden_size), dtype=np.float16)
            prefill_embeds_mask_cpu = np.zeros((num_tokens_static,), dtype=bool)

            mrope_position_ids_cpu = None
            visual_pos_masks_cpu = None
            deepstack_visual_embeds_cpu = None
            if uses_mrope_model:
                mrope_position_ids_cpu = np.zeros((3, num_tokens_static), dtype=np.int32)
                deepstack_indexes = getattr(
                    getattr(self.model.config, "vision_config", None),
                    "deepstack_visual_indexes",
                    None,
                )
                deepstack_layers = len(deepstack_indexes) if deepstack_indexes else 0
                if deepstack_layers:
                    visual_pos_masks_cpu = np.zeros((num_tokens_static,), dtype=bool)
                    deepstack_visual_embeds_cpu = [
                        np.zeros((num_tokens_static, hidden_size), dtype=np.float16) for _ in range(deepstack_layers)
                    ]

            cached = (
                prefill_embeds_cpu,
                prefill_embeds_mask_cpu,
                mrope_position_ids_cpu,
                visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu,
            )
            self._vlm_cpu_buffers[num_tokens_static] = cached

        (
            prefill_embeds_cpu,
            prefill_embeds_mask_cpu,
            mrope_position_ids_cpu,
            visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu,
        ) = cached

        # Clear masks/position ids each step; large embed buffers are overwritten only
        # for the masked regions, and ignored otherwise.
        prefill_embeds_mask_cpu.fill(False)
        if mrope_position_ids_cpu is not None:
            mrope_position_ids_cpu.fill(0)
        if visual_pos_masks_cpu is not None:
            visual_pos_masks_cpu.fill(False)

        return cached

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
        list[dict[int, float] | None],
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
        if row_indices is not None:
            row_indices = np.asarray(row_indices, dtype=np.int32)
            token_ids_window_cpu = self.sequence_buffer.token_ids[row_indices]
            num_computed_tokens_window_cpu = self.sequence_buffer.num_computed_tokens[row_indices]
            page_table_window_cpu = page_table_cpu[row_indices]
            start_index = int(row_indices[0]) if row_indices.size else 0
            packed_rows = True
        else:
            start_index = max(0, int(start_index))
            end_index = start_index + row_count

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

        The scheduler keeps some RUNNING requests resident even when they
        receive zero tokens in the current step. When such rows appear in the
        middle of a window, the execution key can become `(few tokens, many
        requests)`, which is not a real batch shape. This helper preserves the
        common contiguous-prefix fast path and only packs rows when interior
        zero-token gaps are present.
        """
        start_index = max(0, int(start_index))
        stop_index = max(start_index, int(stop_index))

        window_req_ids: list[str | None] = []
        window_scheduled: list[int] = []
        last_positive_offset = -1

        for global_row_index in range(start_index, stop_index):
            rid = self.sequence_buffer.req_ids[global_row_index]
            scheduled = int(scheduled_tokens_by_req.get(rid, 0)) if rid is not None else 0
            window_req_ids.append(rid)
            window_scheduled.append(scheduled)
            if rid is not None and scheduled > 0:
                last_positive_offset = global_row_index - start_index

        if last_positive_offset < 0:
            return np.empty((0,), dtype=np.int32), [], [], stop_index, False

        prefix_stop = last_positive_offset + 1
        has_interior_zero_rows = any(
            rid is None or scheduled <= 0
            for rid, scheduled in zip(window_req_ids[:prefix_stop], window_scheduled[:prefix_stop], strict=False)
        )

        if not has_interior_zero_rows:
            row_indices = np.arange(start_index, start_index + prefix_stop, dtype=np.int32)
            req_ids_window = [typing.cast(str, rid) for rid in window_req_ids[:prefix_stop]]
            scheduled_list = [int(scheduled) for scheduled in window_scheduled[:prefix_stop]]
            return row_indices, req_ids_window, scheduled_list, start_index + prefix_stop, False

        if not allow_sparse_packing:
            row_indices = np.arange(start_index, start_index + prefix_stop, dtype=np.int32)
            req_ids_window = [typing.cast(str | None, rid) for rid in window_req_ids[:prefix_stop]]
            scheduled_list = [int(scheduled) for scheduled in window_scheduled[:prefix_stop]]
            return row_indices, req_ids_window, scheduled_list, start_index + prefix_stop, False

        row_indices_list: list[int] = []
        req_ids_window: list[str] = []
        scheduled_list: list[int] = []
        for offset in range(prefix_stop):
            rid = window_req_ids[offset]
            scheduled = int(window_scheduled[offset])
            if rid is None or scheduled <= 0:
                continue
            row_indices_list.append(start_index + offset)
            req_ids_window.append(rid)
            scheduled_list.append(scheduled)
        return (
            np.asarray(row_indices_list, dtype=np.int32),
            req_ids_window,
            scheduled_list,
            start_index + prefix_stop,
            True,
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

        Notes:
            - `max_model_len` controls the *sequence length* (context window).
            - `max_num_batched_tokens` controls the *per-step* token budget that the
              scheduler will emit in a single forward pass.

            When `max_num_batched_tokens` is provided, compilation is capped to the
            smallest token bucket >= that value (dramatically reducing startup time
            for long-context models).
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

        self._precompile_jitted_helpers(
            reqs_padds=self.active_num_seq_buckets,
            prompt_len_buckets=[min(n, self.max_model_len) for n in num_tokens_paddings],
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
        self.executor_manager.model = None
        self.executor_manager.graphstate = None
        self.executor_manager.graphother = None
        self.executor_manager._model_executor.model = None
        self.executor_manager._sampler_executor.model = None

    def destroy_kv_cache(self) -> None:
        """Destroy the current ragged KV cache to release memory."""
        logger.info("Destroying eSurgeRunner ragged KV cache pages")
        self.executor_manager.kv_pages = None

    def initialize_kv_cache(self) -> None:
        """Reinitialize the ragged KV cache if it has been destroyed."""

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

    def _precompute_vlm_prefill(self, req_state: CachedRequestState) -> None:
        """Precompute prompt embeddings (+ optional mRoPE indices) for VLM requests.

        Some VLM base models compute mRoPE indices via NumPy/data-dependent control-flow
        which is not compatible with JIT/AOT inside the compiled eSurge step. We run
        those parts eagerly here and store host-side arrays for later reuse.
        """
        if req_state.vision_processed:
            return

        uses_mrope = model_uses_mrope(self.model)

        # If raw vision inputs are missing but the request is marked as "has_vision"
        # (e.g. only cached mm_features remain), skip precompute and treat it as processed.
        if req_state.pixel_values is None and req_state.pixel_values_videos is None:
            req_state._vision_processed = True
            return

        if req_state.prefill_inputs_embeds is not None and (
            not uses_mrope or req_state.prefill_position_ids is not None
        ):
            req_state.clear_vision_data()
            return

        prompt_ids = np.asarray(req_state.prompt_token_ids, dtype=np.int32)[None, :]
        input_ids = jnp.asarray(prompt_ids, dtype=jnp.int32)
        attention_mask = jnp.ones(input_ids.shape, dtype=jnp.int32)

        try:
            embed_kwargs: dict[str, typing.Any] = {"attention_mask": attention_mask}
            if req_state.pixel_values is not None:
                embed_kwargs["pixel_values"] = req_state.pixel_values
                if req_state.image_grid_thw is not None:
                    embed_kwargs["image_grid_thw"] = req_state.image_grid_thw
            if req_state.pixel_values_videos is not None:
                embed_kwargs["pixel_values_videos"] = req_state.pixel_values_videos
                if req_state.video_grid_thw is not None:
                    embed_kwargs["video_grid_thw"] = req_state.video_grid_thw

            inputs_embeds, info = self.model.compute_embedding_with_info(input_ids, **embed_kwargs)
        except Exception as exc:
            logger.warning(f"VLM precompute failed for req_id={req_state.req_id}: {exc}")
            return

        # Store host-side views (keeps compiled step free of vision preprocessing).
        embeds_host = np.asarray(jax.device_get(inputs_embeds))
        req_state.prefill_inputs_embeds = embeds_host[0]

        if getattr(info, "position_ids", None) is not None:
            pos_host = np.asarray(jax.device_get(info.position_ids))
            if pos_host.ndim == 3:
                pos_host = pos_host[:, 0, :]
            req_state.prefill_position_ids = pos_host.astype(np.int32, copy=False)

        if getattr(info, "rope_deltas", None) is not None:
            req_state.prefill_rope_deltas = np.asarray(jax.device_get(info.rope_deltas)).astype(np.int32, copy=False)

        if getattr(info, "visual_pos_masks", None) is not None:
            mask_host = np.asarray(jax.device_get(info.visual_pos_masks))
            if mask_host.ndim == 2:
                mask_host = mask_host[0]
            req_state.prefill_visual_pos_masks = mask_host.astype(bool, copy=False)

        if getattr(info, "deepstack_visual_embeds", None) is not None:
            ds_list = []
            for arr in info.deepstack_visual_embeds:
                ds_list.append(np.asarray(jax.device_get(arr)))
            req_state.prefill_deepstack_visual_embeds = ds_list

        # Raw vision tensors are no longer needed once embeddings are cached.
        req_state.clear_vision_data()

    def _update_states(self, scheduler_output: SchedulerOutput) -> bool:
        """Update internal states based on scheduler output.

        Synchronizes the runner's internal state with the scheduler's decisions.
        Handles request lifecycle: adding new requests, removing finished ones,
        updating cached requests, and managing the sequence buffer.

        State Updates:
            1. Remove finished requests from tracking
            2. Remove unscheduled requests from buffer
            3. Add new requests with their metadata
            4. Update cached request states
            5. Reorganize sequence buffer for efficiency

        Args:
            scheduler_output: Contains request scheduling decisions including:
                - finished_req_ids: Requests that completed
                - scheduled_new_reqs: New requests to add
                - scheduled_cached_reqs: Existing requests to update
                - num_scheduled_tokens: Tokens to generate per request

        Returns:
            True if state changed (requests added/removed), indicating
            potential buffer reorganization. False if no changes occurred.

        Side Effects:
            - Updates self.requests dictionary
            - Modifies sequence buffer contents
            - May trigger buffer condensation

        Note:
            This method is called at the beginning of each execution cycle
            to ensure the runner's state matches the scheduler's decisions.
        """
        dp_size = int(getattr(self.metadata, "data_parallel_size", 1) or 1)
        pages_per_shard_opt = pages_per_dp_shard(int(getattr(self.metadata, "num_pages", 0) or 0), dp_size)
        use_dp_local_rows = (
            dp_size > 1
            and int(self.sequence_buffer.max_num_reqs) > 0
            and int(self.sequence_buffer.max_num_reqs) % dp_size == 0
            and pages_per_shard_opt is not None
        )
        rows_per_shard = int(self.sequence_buffer.max_num_reqs) // dp_size if use_dp_local_rows else 0
        pages_per_shard = int(pages_per_shard_opt or 0) if use_dp_local_rows else 0

        def infer_req_shard(page_ids: tuple[list[int], ...]) -> int | None:
            """Infer the DP shard index that owns a request based on its page IDs.

            Examines page IDs across all cache groups and returns the shard
            index if all non-null pages belong to the same shard. Returns
            None if DP-local rows are disabled or pages span multiple shards.
            """
            if not use_dp_local_rows or pages_per_shard <= 0:
                return None
            inferred: int | None = None
            for group_ids in page_ids:
                for pid in group_ids:
                    # 0 is reserved for null/padding page in page pool.
                    if int(pid) <= 0:
                        continue
                    shard = dp_shard_for_page_id(int(pid), pages_per_shard, dp_size)
                    if shard is None:
                        continue
                    if inferred is None:
                        inferred = shard
                    elif inferred != shard:
                        return None
            return inferred

        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # 2) Remove finished from sequence buffer (functional)
        removed_req_indices: list[int] = []
        removed_req_index_by_id: dict[str, int] = {}
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.sequence_buffer.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # 3) Remove preempted requests from buffer.
        # Only remove requests the scheduler explicitly preempted (evicted from
        # running to waiting). Running requests that were merely skipped due to
        # token budget exhaustion still hold valid rows and pages — removing them
        # would force re-insertion next cycle and trigger "No free sequence row
        # in target DP shard" errors when shard rows are full.
        for req_id in scheduler_output.preempted_req_ids:
            req_index = self.sequence_buffer.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)
                removed_req_index_by_id[req_id] = req_index

        # 3b) Clear recurrent/SSM state for freed slots so the next request
        # assigned to the same slot starts from a clean state.
        if removed_req_indices:
            self.executor_manager.clear_recurrent_slots(removed_req_indices)

        # 4) Add new requests to tracking
        req_ids_to_add: list[str] = []
        for new_req_data in scheduler_output.scheduled_new_reqs:
            if new_req_data.sampling_params is None:
                raise ValueError("Pooling not supported in TPU")
            req_id = new_req_data.req_id
            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=new_req_data.sampling_params,
                generator=None,
                page_ids=new_req_data.page_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                # Vision-language model data
                pixel_values=new_req_data.pixel_values,
                image_grid_thw=new_req_data.image_grid_thw,
                pixel_values_videos=new_req_data.pixel_values_videos,
                video_grid_thw=new_req_data.video_grid_thw,
                mm_features=new_req_data.mm_features,
            )
            req_ids_to_add.append(req_id)

        # 5) Update cached requests and page tables
        req_data = scheduler_output.scheduled_cached_reqs
        upd_req_indices: list[int] = []
        upd_num_computed_vals: list[int] = []
        batched_page_rows: list[tuple[int, tuple[list[int], ...]]] = []

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests.get(req_id)
            if req_state is None:
                continue

            nct = req_data.num_computed_tokens[i]
            new_page_ids = req_data.new_page_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            req_state.num_computed_tokens = nct
            if not resumed_from_preemption:
                for page_ids, new_ids in zip(req_state.page_ids, new_page_ids, strict=False):
                    page_ids.extend(new_ids)
            else:
                req_state.page_ids = new_page_ids

            req_index = self.sequence_buffer.req_id_to_index.get(req_id)
            if req_index is None:
                req_ids_to_add.append(req_id)
                continue

            upd_req_indices.append(req_index)
            upd_num_computed_vals.append(int(nct))
            if resumed_from_preemption:
                # Resumed requests may provide a full replacement page table.
                self.sequence_buffer.page_table.add_row(new_page_ids, req_index)
            else:
                # Running requests report only incremental page allocations.
                if any(len(ids) for ids in new_page_ids):
                    batched_page_rows.append((req_index, new_page_ids))

        if upd_req_indices:
            # num_computed_tokens is now a NumPy array, use standard indexing
            idx_arr = np.array(upd_req_indices, dtype=np.int32)
            val_arr = np.array(upd_num_computed_vals, dtype=np.int32)
            self.sequence_buffer.num_computed_tokens[idx_arr] = val_arr

        if batched_page_rows:
            indices = [ix for ix, _ in batched_page_rows]
            pages_per_req = [ids for _, ids in batched_page_rows]
            self.sequence_buffer.page_table.append_rows_batch(pages_per_req, indices)

        # 6) Add new / reinserted requests
        # Prefer stable index reuse (same request index when possible), and under
        # DP-local page sharding, try to keep request rows in the shard range that
        # matches their current page IDs.
        removed_pool = set(removed_req_indices)

        def _find_reuse_index_in_shard(shard_idx: int) -> int | None:
            if not use_dp_local_rows:
                return None
            lo = int(shard_idx) * rows_per_shard
            hi = lo + rows_per_shard

            shard_removed = [ix for ix in removed_pool if lo <= int(ix) < hi]
            if shard_removed:
                return max(shard_removed)

            req_slots = self.sequence_buffer.req_ids
            for ix in range(lo, hi):
                if ix >= len(req_slots) or req_slots[ix] is None:
                    return ix
            return None

        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            reuse_index = removed_req_index_by_id.pop(req_id, None)
            if reuse_index is not None and reuse_index not in removed_pool:
                reuse_index = None

            target_shard = infer_req_shard(req_state.page_ids)
            if target_shard is not None and use_dp_local_rows:
                lo = target_shard * rows_per_shard
                hi = lo + rows_per_shard

                if reuse_index is not None and not (lo <= int(reuse_index) < hi):
                    logger.warning(
                        "Dropping out-of-shard row reuse for req %s: reuse_index=%s target_shard=%s range=[%s,%s).",
                        req_id,
                        reuse_index,
                        target_shard,
                        lo,
                        hi,
                    )
                    reuse_index = None

                if reuse_index is None:
                    reuse_index = _find_reuse_index_in_shard(target_shard)

                # Preserve DP-local block-table invariants: a request that already
                # owns shard-local pages must never be inserted into a different
                # row shard. If no row is available in this shard, surface a hard
                # error so scheduler/accounting can be fixed.
                if reuse_index is None:
                    raise RuntimeError(
                        "No free sequence row available in target DP shard for request insertion. "
                        f"req_id={req_id} shard={target_shard} rows_per_shard={rows_per_shard} "
                        f"removed_pool_size={len(removed_pool)}."
                    )

            if reuse_index is None and removed_pool:
                reuse_index = max(removed_pool)

            if reuse_index is not None:
                removed_pool.discard(reuse_index)
            self.sequence_buffer.add_request(req_state, reuse_index)

        # 7) Condense to remove holes
        if removed_pool and not use_dp_local_rows:
            self.sequence_buffer.condense(sorted(removed_pool))

        # Drop cached VLM prompt helpers once prefill is complete to free host RAM.
        for req_state in self.requests.values():
            if (
                req_state.prefill_inputs_embeds is not None
                and req_state.num_computed_tokens >= req_state.num_prompt_tokens
            ):
                req_state.prefill_inputs_embeds = None
                req_state.prefill_position_ids = None
                req_state.prefill_visual_pos_masks = None
                req_state.prefill_deepstack_visual_embeds = None

        has_changes = len(scheduler_output.preempted_req_ids) > 0 or len(req_ids_to_add) > 0
        return has_changes

    def _modify_prev_results(self) -> None:
        """Apply previous iteration's tokens to sequence buffer.

        This method is called at the beginning of each iteration when async
        scheduling is enabled. It retrieves the tokens that were sampled
        asynchronously in the previous iteration and applies them to the
        sequence buffer.

        The method blocks until the async token transfer is complete, then
        updates the token_ids array and request output_token_ids lists.

        Note:
            This method should only be called when self._pre_async_results is not None.
        """
        if self._pre_async_results is None:
            return

        pre_windows = self._pre_async_results.windows
        pre_request_seq_lens = self._pre_async_results.request_seq_lens

        valid_sampled_token_ids: list[np.ndarray] = []
        for window in pre_windows:
            next_tokens_cpu = np.asarray(window.sampled_token_ids)
            for row_pos, is_valid in zip(window.row_positions, window.valid_mask, strict=False):
                if not is_valid:
                    valid_sampled_token_ids.append(np.array([], dtype=np.int32))
                    continue
                valid_sampled_token_ids.append(np.array([int(next_tokens_cpu[row_pos])], dtype=np.int32))

        # Apply tokens to sequence buffer
        for pre_req_idx, _, req_state, _ in pre_request_seq_lens:
            sampled_ids = valid_sampled_token_ids[pre_req_idx]
            if len(sampled_ids) == 0:
                continue

            # Check if request is still active
            req_id = req_state.req_id
            if req_id not in self.sequence_buffer.req_id_to_index or req_id not in self.requests:
                continue

            req_idx = self.sequence_buffer.req_id_to_index[req_id]
            if req_state is not self.requests[req_id]:
                raise RuntimeError("Request state mismatch")

            # Update token_ids array (replace placeholder)
            end_idx = self.sequence_buffer.num_tokens_no_spec[req_idx]
            start_idx = end_idx - 1
            if end_idx > self.max_model_len:
                raise ValueError(f"Token count {end_idx} exceeds max_model_len {self.max_model_len}")

            self.sequence_buffer.token_ids[req_idx, start_idx:end_idx] = sampled_ids
            # Replace placeholder in output_token_ids
            req_state.output_token_ids[-1] = int(sampled_ids[-1])

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

        for out_idx, seq_row_idx, req_state, _ in request_seq_lens:
            if out_idx in discard_set:
                continue

            start_idx = self.sequence_buffer.num_tokens_no_spec[seq_row_idx]
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

        Partitions the request buffer so all decode requests (single token,
        with computed tokens > 0) appear before prefill requests. This ordering
        matches the TPU runner behavior and enables optimized v3 attention
        with request distribution.

        Args:
            scheduler_output: Used to determine scheduled tokens per request.

        Side Effects:
            - Modifies sequence_buffer ordering via swap_states().
        """
        i, j = 0, self.sequence_buffer.num_reqs - 1
        while i < j:
            i_req_id = self.sequence_buffer.req_ids[i]
            j_req_id = self.sequence_buffer.req_ids[j]
            if i_req_id is None or j_req_id is None:
                break

            i_is_decode = (
                scheduler_output.num_scheduled_tokens.get(i_req_id, 0) == 1
                and self.sequence_buffer.num_computed_tokens[i] > 0
            )
            j_is_decode = (
                scheduler_output.num_scheduled_tokens.get(j_req_id, 0) == 1
                and self.sequence_buffer.num_computed_tokens[j] > 0
            )

            if i_is_decode:
                i += 1
            elif not j_is_decode:
                j -= 1
            else:
                # Swap to move a decode request forward.
                self.sequence_buffer.swap_states(i, j)
                i += 1
                j -= 1

    def _reorder_decode_first_per_shard(
        self,
        scheduler_output: SchedulerOutput,
        dp_size: int,
    ) -> None:
        """Reorder decode requests first within each DP shard's row range.

        Unlike _reorder_decode_first which reorders across the entire buffer
        (and would move requests across shard boundaries), this method
        reorders decode-first independently within each shard's contiguous
        row range: [shard * rows_per_shard, (shard+1) * rows_per_shard).

        This preserves DP-local row placement while giving each shard's
        rows the decode-first ordering that the v3 attention kernel expects.

        Args:
            scheduler_output: Used to determine scheduled tokens per request.
            dp_size: Number of data-parallel shards.
        """
        # Use max_num_reqs (not num_slots) for shard boundaries to match
        # _update_states and the validation in batch_preparer, which both
        # partition rows based on the fixed max_num_reqs capacity.
        max_reqs = self.sequence_buffer.max_num_reqs
        if max_reqs <= 1 or dp_size <= 1:
            return
        rows_per_shard = max_reqs // dp_size
        if rows_per_shard <= 1 or max_reqs % dp_size != 0:
            return

        num_slots = self.sequence_buffer.num_slots
        for shard in range(dp_size):
            lo = shard * rows_per_shard
            hi = min(lo + rows_per_shard, num_slots)

            # 1) Compact holes (None slots) to the end of the shard range.
            #    This ensures the attention kernel never encounters a 0-token
            #    row in the middle of its processing range.
            self.sequence_buffer.compact_holes_in_range(lo, hi)

            # 2) Decode-first partitioning on the compacted (hole-free) prefix.
            #    Find the boundary between non-None rows and holes.
            shard_end = hi
            while shard_end > lo and (
                shard_end - 1 >= len(self.sequence_buffer.req_ids) or self.sequence_buffer.req_ids[shard_end - 1] is None
            ):
                shard_end -= 1

            i, j = lo, shard_end - 1
            while i < j:
                i_req_id = self.sequence_buffer.req_ids[i]
                j_req_id = self.sequence_buffer.req_ids[j]

                # Guard against empty slots that survived compaction
                # (e.g. when prompt count < max_num_seqs).
                if i_req_id is None:
                    break  # no more populated slots from the left
                if j_req_id is None:
                    j -= 1
                    continue

                i_is_decode = (
                    scheduler_output.num_scheduled_tokens.get(i_req_id, 0) == 1
                    and self.sequence_buffer.num_computed_tokens[i] > 0
                )
                j_is_decode = (
                    scheduler_output.num_scheduled_tokens.get(j_req_id, 0) == 1
                    and self.sequence_buffer.num_computed_tokens[j] > 0
                )

                if i_is_decode:
                    i += 1
                elif not j_is_decode:
                    j -= 1
                else:
                    self.sequence_buffer.swap_states(i, j)
                    i += 1
                    j -= 1

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
        self._update_states(scheduler_output)
        updating_states_time = time.time() - updating_states_start

        # Apply previous async results if available
        prev_async_start = time.time()
        if self._pre_async_results is not None:
            self._modify_prev_results()
            self._pre_async_results = None  # Clear after applying
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

        if not scheduler_output.total_num_scheduled_tokens:
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

        needs_async_output = return_async_output or scheduler_output.async_scheduling
        start_index = 0
        total_step_time = 0.0
        total_post_proc_time = 0.0

        req_ids_all: list[str] = []
        sampled_token_ids_all: list[list[int]] = []
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
        total_execute_overhead_time = 0.0
        total_pp_stage_dispatch_time = 0.0
        total_pp_queue_wait_time = 0.0
        total_pp_stage_launches = 0
        total_pp_stage_compute_time = 0.0
        total_pp_stage_max_time = 0.0
        pp_stage_times_by_index: dict[int, float] = {}
        total_runner_host_time = 0.0
        total_d2h_time = 0.0
        token_buckets_used: set[int] = set()
        req_buckets_used: set[int] = set()
        request_seq_lens: list[tuple[int, int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices: list[int] = []

        cfg = getattr(self.model, "config", None)
        task_type = getattr(self.model, "_task_type", None)
        is_vlm_model = task_type == "image-text-to-text" or (
            cfg is not None
            and (getattr(cfg, "image_token_id", None) is not None or getattr(cfg, "video_token_id", None) is not None)
            and callable(getattr(self.model, "get_image_features", None))
        )
        uses_mrope_model = model_uses_mrope(self.model)

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

            total_scheduled = sum(scheduled_list)
            idx = bisect_left(self.num_tokens_paddings, total_scheduled)
            if idx >= len(self.num_tokens_paddings):
                idx = len(self.num_tokens_paddings) - 1
            num_tokens_static = int(self.num_tokens_paddings[idx])

            # Select optimal bucket for current batch size
            # This determines which compiled function to use
            current_bucket = self._get_current_bucket(num_reqs)
            padded_num_reqs = current_bucket  # Use bucket size for compilation lookup

            scheduled_full_cpu = self._scheduled_full_cpu
            active_mask_full_cpu = self._active_mask_full_cpu
            if num_reqs > 0:
                # Keep scheduled and active_mask as CPU arrays
                scheduled_full_cpu = self._scheduled_full_cpu
                scheduled_full_cpu.fill(0)
                scheduled_full_cpu[: len(scheduled_list)] = scheduled_list

                # Packed view of the per-request target lengths for the current window.
                # Avoid per-step dict lookups; SequenceBuffer keeps this aligned with its ordering.
                req_num_tokens_np = self._req_num_tokens_cpu
                req_num_tokens_np.fill(0)
                req_num_tokens_np[:num_reqs] = self.sequence_buffer.num_tokens[window_row_indices]

                active_mask_full_cpu = self._active_mask_full_cpu
                active_mask_full_cpu.fill(False)
                for i, rid in enumerate(req_ids_window):
                    if rid is not None:
                        active_mask_full_cpu[i] = True

                window_row_indices_cpu = self._window_row_indices_cpu
                window_row_indices_cpu.fill(0)
                window_row_indices_cpu[:num_reqs] = window_row_indices

                if jax.process_count() > 1:
                    req_num_tokens_np = multihost_utils.broadcast_one_to_all(req_num_tokens_np)
                self.req_num_tokens_full_buf = jax.device_put(req_num_tokens_np, self._empty_sharding)

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
                    n = int(scheduled_list[req_idx])
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
                        if req_id_dbg is None or int(scheduled_list[local_req_idx]) <= 0:
                            continue
                        global_row_index = int(window_row_indices[local_req_idx])
                        seq_len = int(self.sequence_buffer.num_computed_tokens[global_row_index]) + int(
                            scheduled_list[local_req_idx]
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
                                int(scheduled_list[local_req_idx]),
                                int(self.sequence_buffer.num_computed_tokens[global_row_index]),
                            )
                            break

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
                req_num_tokens_full=self.req_num_tokens_full_buf,
                active_mask_full_cpu=active_mask_full_cpu,
                window_row_indices_cpu=window_row_indices_cpu,
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
                mrope_position_ids_cpu=mrope_position_ids_cpu,
                prefill_embeds_cpu=prefill_embeds_cpu,
                prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
                visual_pos_masks_cpu=visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
                wait_for_outputs=not needs_async_output,
            )

            # account for device time (blocking already happened inside execute())
            total_step_time += time.time() - step_start
            num_windows += 1
            total_exec_time += float(window_metrics.get("exec_time", 0.0))
            total_sample_time += float(window_metrics.get("sample_time", 0.0))
            total_prep_time += float(window_metrics.get("prep_time", 0.0))
            total_prep_host_time += float(window_metrics.get("prep_host_time", 0.0))
            total_prep_put_time += float(window_metrics.get("prep_put_time", 0.0))
            total_prep_extra_put_time += float(window_metrics.get("prep_extra_put_time", 0.0))
            total_execute_overhead_time += float(window_metrics.get("execute_overhead_time", 0.0))
            total_pp_stage_dispatch_time += float(window_metrics.get("pp_stage_dispatch_time", 0.0))
            total_pp_queue_wait_time += float(window_metrics.get("pp_queue_wait_time", 0.0))
            total_pp_stage_launches += int(window_metrics.get("pp_stage_launches", 0))
            total_pp_stage_compute_time += float(window_metrics.get("pp_stage_compute_time", 0.0))
            total_pp_stage_max_time = max(
                total_pp_stage_max_time,
                float(window_metrics.get("pp_stage_max_time", 0.0)),
            )
            for stage_idx in range(8):
                key = f"pp_stage_{stage_idx}_time"
                if key in window_metrics:
                    pp_stage_times_by_index[stage_idx] = pp_stage_times_by_index.get(stage_idx, 0.0) + float(
                        window_metrics[key]
                    )
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
                        request_seq_lens.append((out_idx, req_idx, req_state, seq_len))
                    else:
                        discard_sampled_tokens_req_indices.append(out_idx)
                elif return_async_output:
                    sync_finalize_entries.append((req_state, req_idx, seq_len))

            if needs_async_output:
                d2h_start = time.time()
                row_positions = [row_pos for row_pos, *_rest in window_entries]
                async_windows.append(
                    AsyncWindowResult(
                        req_ids=[rid for _, rid, *_rest in window_entries],
                        row_positions=row_positions,
                        sampled_token_ids=jax.copy_to_host_async(out_tokens_win[:num_reqs]),
                        valid_mask=[is_valid for *_, is_valid in window_entries],
                        token_logprobs=(
                            jax.copy_to_host_async(_logits[:num_reqs])
                            if self.enable_sampler_metrics and _logits is not None
                            else None
                        ),
                    )
                )
                total_d2h_time += time.time() - d2h_start
                total_post_proc_time += time.time() - up_wtime
                start_index = next_start_index
                continue

            d2h_start = time.time()
            tokens_np = np.asarray(out_tokens_win)
            _logits_maybe: typing.Any | None = _logits
            logits_np = np.asarray(_logits_maybe) if self.enable_sampler_metrics and _logits_maybe is not None else None
            total_d2h_time += time.time() - d2h_start

            for row_pos, rid, req_state, req_idx, seq_len, is_valid in window_entries:
                if not is_valid:
                    sampled_token_ids_all.append([])
                    continue

                tid = int(tokens_np[row_pos])
                if req_state is not None and seq_len is not None:
                    if req_idx is not None and 0 <= seq_len < self.max_model_len:
                        self.sequence_buffer.token_ids[req_idx, seq_len] = tid
                    sampled_token_ids_all.append([tid])
                    req_state.output_token_ids.append(tid)
                else:
                    sampled_token_ids_all.append([tid])

                if self.enable_sampler_metrics and logits_np is not None and row_pos < logits_np.shape[0]:
                    try:
                        token_logprobs[rid] = logits_np[row_pos]
                    except Exception:
                        pass

            total_post_proc_time += time.time() - up_wtime

            start_index = next_start_index

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
                self._pre_async_results = AsyncPreResults(
                    windows=async_windows,
                    request_seq_lens=request_seq_lens,
                )

            def _finalize_sync_runner_state(sampled_token_ids: list[list[int]]) -> None:
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
                    spec_token_ids=None,
                    logprobs=None,
                    prompt_logprobs_dict={rid: None for rid in req_ids_all},
                    finished_sending=None,
                    finished_recving=None,
                    token_logprobs=None,
                ),
                windows=async_windows,
                finalize=None if scheduler_output.async_scheduling else _finalize_sync_runner_state,
            )
            if return_async_output:
                final_output = async_output
            else:
                d2h_finalize_start = time.time()
                resolved_output = async_output.get_output()
                total_d2h_time += time.time() - d2h_finalize_start
                final_output = resolved_output
                token_logprobs = resolved_output.token_logprobs or token_logprobs
        else:
            final_output = ModelRunnerOutput(
                req_ids=req_ids_all,
                req_id_to_index=req_id_to_out_index,
                req_id_to_row_index=req_id_to_row_index,
                sampled_token_ids=sampled_token_ids_all,
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={rid: None for rid in req_ids_all},
                finished_sending=None,
                finished_recving=None,
                token_logprobs=token_logprobs or None,
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
            + total_d2h_time
            + total_post_proc_time
            + total_prep_time
            + total_exec_time
            + total_sample_time
            + total_execute_overhead_time
            + step_gap_time
            + metrics_time
        )
        misc_time = max(0.0, misc_time)

        prep_detail = ""
        if (total_prep_host_time + total_prep_put_time + total_prep_extra_put_time) > 0:
            prep_detail = (
                f"(host={total_prep_host_time * 1e3:.2f}ms put={total_prep_put_time * 1e3:.2f}ms "
                f"extra={total_prep_extra_put_time * 1e3:.2f}ms) "
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
            f"runner={total_runner_host_time * 1e3:.2f}ms d2h={total_d2h_time * 1e3:.2f}ms "
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
        return self._execute_model_impl(scheduler_output)

    def execute_model_async(self, scheduler_output: SchedulerOutput) -> _AsyncExecutionHandle:
        """Dispatch model work and defer the host-side token materialization.

        TPU/JAX dispatch is already asynchronous on the calling thread. This
        method exploits that by keeping execution on the scheduler thread,
        returning an async handle once the device work and host copies have been
        queued, and letting the lifecycle loop do scheduler prefetch work before
        calling wait_for_execution().
        """
        return self._execute_model_impl(scheduler_output, return_async_output=True)

    def initialize_async_executor(self) -> None:
        """Retained for API compatibility.

        Older overlap code used a background ThreadPoolExecutor here. TPU/JAX
        proved unreliable when the compiled step ran on a different Python
        thread, so overlap now uses same-thread async handles instead.
        """
        if self._executor is not None:
            logger.debug("Shutting down legacy async executor")
            self._executor.shutdown(wait=True)
            self._executor = None
        logger.debug("Using same-thread async execution handles for overlap")

    def reset_state(self) -> None:
        """Clear sequence state and request bookkeeping.

        Useful when pausing or resetting the runner to ensure no stale pages
        or request metadata linger between sessions.
        """
        self.requests.clear()
        self.sequence_buffer.clear()
        self._pre_async_results = None

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
        """Cleanup resources including async executor if present."""
        if self._executor is not None:
            logger.debug("Shutting down async executor")
            self._executor.shutdown(wait=True)
            self._executor = None
        if getattr(self, "executor_manager", None) is not None:
            self.executor_manager.shutdown()
