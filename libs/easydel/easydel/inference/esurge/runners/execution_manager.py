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

"""Execution manager for high-performance model inference with fused step functions.

This module implements the ExecutionManager class, which handles compilation, caching,
and execution of fused inference steps. The manager pre-compiles functions for multiple
input configurations to eliminate runtime compilation overhead during serving.

Architecture:
    The manager uses a fused execution model where a single JIT-compiled function
    combines four sequential operations:

    1. Input preparation: Token gathering and position calculation
    2. Model forward pass: Transformer execution with paged attention
    3. Token sampling: Stochastic sampling with temperature/top-k/top-p
    4. State updates: Token buffer updates and sequence tracking

    This fusion minimizes host-device communication (single dispatch per step) and
    maximizes kernel fusion opportunities within JAX/XLA.

Compilation Modes:
    - AOT (Ahead-of-Time): Pre-compiles all configurations using lower().compile()
      for predictable latency and minimal warmup. Default for production.
    - JIT (Just-in-Time): Defers compilation to first execution. Faster initial
      setup but unpredictable first-step latency.

Performance Characteristics:
    - Single host-device round-trip per inference step
    - Automatic kernel fusion via XLA compiler
    - Bucketed compilation: O(log N) unique compilations for N request sizes
    - Compiled variants are retained until the cache is explicitly cleared

Example:
    >>> from easydel.inference.esurge.runners import ExecutionManager
    >>> executor = ExecutionManager(
    ...     model=model,
    ...     mesh=jax.sharding.Mesh(devices, ('dp', 'tp')),
    ...     kv_pages=cache,
    ...     use_aot_forward=True,
    ... )
    >>> executor.compile(
    ...     num_tokens_paddings=[128, 256, 512, 1024],
    ...     num_reqs_max_model_len=16,
    ...     max_pages_per_req=64,
    ...     max_num_reqs=32,
    ...     metadata=cache_metadata,
    ... )
    >>> result = executor.execute(
    ...     num_tokens=256,
    ...     device_state=state,
    ...     scheduled_full=scheduled,
    ...     req_num_tokens_full=req_tokens,
    ...     active_mask_full=active_mask,
    ...     input_ids_buf=input_buf,
    ...     position_ids_buf=pos_buf,
    ...     padded_num_reqs=16,
    ... )
"""

from __future__ import annotations

import time
import typing
import typing as tp
from functools import lru_cache

import jax
import numpy
import spectrax as spx
from eformer.loggings import get_logger
from ejkernel.ops import forward_autotune_only  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from easydel.axis import resolve_attention_data_parallel_axis
from easydel.caching import (
    HybridCache,
    ParallelHybridCacheView,
    RaggedPagesCache,
    RaggedPagesCacheConfig,
    RecurrentCacheView,
    UnifiedAttentionCache,
    UnifiedAttentionCacheConfig,
)
from easydel.infra.sharding import replicated_named_sharding
from easydel.layers.quantization import TurboQuantConfig
from easydel.utils import flags

from ..utils import model_uses_mrope
from .async_types import DeviceInputTokenHandoff
from .execution_types import ModelStepOutputs, StepFunctionInputs
from .executors import BatchMetadataPreparer, ModelStepExecutor, SamplerExecutor, SamplerRuntime
from .executors.compile_orchestrator import CompileOrchestrator, _tree_hash, _tree_hash_diff
from .executors.pipeline_microbatch import PipelineMicrobatchExecutor
from .executors.sampler_executor import (
    _cpu_vector_size,
    _greedy_argmax_tokens,
    _padded_cpu_window,
)
from .pipeline_plan import PipelineInferencePlan, metadata_num_pages, set_metadata_num_pages
from .sequence_buffer import SequenceBuffer

DEBUG_MODE = False

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule
    from easydel.infra.etils import MpMdSchedulers

logger = get_logger("eSurge-ExecutionManager")

# Syncing inputs after host->device metadata transfer makes `prep_time` more accurate,
# but it adds a device round-trip that hurts throughput. Keep it opt-in.
SYNC_INPUTS_FOR_TIMING = flags.get_bool(flags.EASYDEL_SYNC_INPUTS_FOR_TIMING)


def _zero_rows_body(arrays: list[jax.Array], keep_mask: jax.Array) -> list[jax.Array]:
    """Zero masked rows of every array in ONE fused dispatch.

    Eager per-array ``jnp.where`` in a python loop over a hybrid model's ~100
    recurrent leaves costs one host dispatch each per slot-clear event; one
    jitted list call collapses that to a single dispatch.
    """
    return [jnp.where(keep_mask.reshape(-1, *([1] * (array.ndim - 1))), array, 0) for array in arrays]


def _permute_rows_body(arrays: list[jax.Array], gather_idx: jax.Array, keep_mask: jax.Array) -> list[jax.Array]:
    """Row-gather + keep-mask every array in ONE fused dispatch (see _zero_rows_body)."""
    out = []
    for array in arrays:
        gathered = array[gather_idx]
        out.append(jnp.where(keep_mask.reshape(-1, *([1] * (array.ndim - 1))), gathered, 0).astype(array.dtype))
    return out


@lru_cache(maxsize=32)
def _jit_zero_rows(output_shardings: tuple[jax.sharding.Sharding, ...]):
    """Compile the slot-clear transform pinned to the live cache layout.

    Pinning ``out_shardings`` to the leaves' own shardings keeps row
    maintenance inside the layout the AOT model executable captured. The
    row-gather in :func:`_jit_permute_rows` otherwise lets the partitioner
    all-gather its way to a replicated ``P()`` result, which the next
    compiled model call rejects. The clear path happens to preserve the
    layout on its own, but pinning both is uniform and costs nothing
    measurable.

    Args:
        output_shardings: Per-leaf shardings, in the order the leaves are
            passed to the returned function.

    Returns:
        A jitted callable ``(arrays, keep_mask) -> arrays``.
    """
    return jax.jit(_zero_rows_body, out_shardings=list(output_shardings))


@lru_cache(maxsize=1024)
def _jit_permute_rows(output_shardings: tuple[jax.sharding.Sharding, ...]):
    """Compile the slot-permute transform pinned to the live cache layout.

    Args:
        output_shardings: Per-leaf shardings, in the order the leaves are
            passed to the returned function.

    Returns:
        A jitted callable ``(arrays, gather_idx, keep_mask) -> arrays``.
    """
    return jax.jit(_permute_rows_body, out_shardings=list(output_shardings))


@lru_cache(maxsize=128)
def _jit_slot_sidecar_rows(output_shardings, fill_values):
    """Transform auxiliary slot leaves together while preserving AOT shardings."""

    def transform(arrays, gather, keep):
        return [
            jnp.where(keep.reshape(-1, *([1] * (array.ndim - 1))), array[gather], fill)
            for array, fill in zip(arrays, fill_values, strict=True)
        ]

    return jax.jit(transform, out_shardings=list(output_shardings))


def _transform_recurrent_sidecars(views, make_indices):
    """Apply the native slot mapping to positions and Qwen PLE continuation.

    Paged attention leaves are intentionally excluded: their leading dimension
    indexes physical pages, not recurrent request slots. Segment padding uses
    -1 rather than zero, matching the PLE cache initialization contract.
    """
    by_pool = {}
    for index, view in enumerate(views):
        rec = view.recurrent if isinstance(view, ParallelHybridCacheView) else view
        if not isinstance(rec, RecurrentCacheView):
            continue
        for name, fill in (
            ("positions", 0),
            ("ple_conv_state", 0),
            ("ple_token_context", 0),
            ("ple_segment_context", -1),
        ):
            array = getattr(rec, name, None)
            if array is not None:
                by_pool.setdefault(int(array.shape[0]), []).append((index, name, array, fill))
    updates = {}
    for n_slots, entries in by_pool.items():
        indices = make_indices(n_slots)
        if indices is None:
            continue
        gather, keep = indices
        arrays = [entry[2] for entry in entries]
        transform = _jit_slot_sidecar_rows(
            tuple(array.sharding for array in arrays), tuple(entry[3] for entry in entries)
        )
        result = transform(arrays, jnp.asarray(gather, jnp.int32), jnp.asarray(keep, bool))
        for (index, name, _array, _fill), value in zip(entries, result, strict=True):
            updates.setdefault(index, {})[name] = value
    for index, values in updates.items():
        view = views[index]
        if isinstance(view, ParallelHybridCacheView):
            # The wrapper's post-init treats existing mirrored positions as
            # overrides, so update that handle with the inner recurrent view.
            mirrors = {"positions": values["positions"]} if "positions" in values else {}
            views[index] = view.replace(recurrent=view.recurrent.replace(**values), **mirrors)
        else:
            views[index] = view.replace(**values)
    return bool(updates)


class ExecutionManager:
    """Compilation and execution manager for fused inference step functions.

    The ExecutionManager pre-compiles and caches fused step functions for multiple
    input configurations, enabling low-latency serving without runtime compilation.
    It uses bucketed compilation (powers of 2) to reduce the number of unique
    variants while maintaining good hardware utilization.

    Architecture:
        The manager splits the model into (graphdef, graphstate, graphother) for
        efficient functional transformations. The graphstate (weights) can be
        updated without recompilation. Compiled functions are retained until
        the cache is explicitly cleared.

    Compilation Strategy:
        Request counts are bucketed into powers of 2 (up to min_input_pad, then
        nearest power of 2 above). Token counts use explicit padding values provided
        during compile(). This produces O(log N * M) compilations for N request
        sizes and M token configurations.

    Attributes:
        model: EasyDeL model instance (EasyDeLBaseModule).
        mesh: JAX sharding mesh for distributed execution across devices.
        kv_pages: Paged KV cache storage (RaggedPagesCache).
        use_aot_forward: If True, use AOT compilation via lower().compile().
            If False, use JIT compilation on first call. Default: True.
        min_input_pad: Minimum request count padding for bucketing. Default: 8.
        max_model_len: Maximum sequence length supported by model.
        max_num_reqs: Maximum concurrent requests.
        max_num_tokens: Maximum tokens per batch (defaults to max_model_len).
        metadata: KV cache config (ragged pages or unified attention).
        graphdef: Model graph definition (static structure).
        graphstate: Model graph state (weights, device-resident).
        graphother: Auxiliary model state (buffers, etc.).
        rng_key: JAX random key for sampling, threaded through steps.

    Private Attributes:
        _batch_preparer: CPU-first batch metadata builder and async transfer helper.
        _model_executor: Model-step executor with compiled-variant cache.
        _sampler_executor: Sampler executor with compiled-variant cache.
        _debug_baselines: Hash baselines for debugging recompilations.
        _empty_sharding: Default sharding (replicated across mesh).

    Example:
        >>> # Initialize manager
        >>> executor = ExecutionManager(
        ...     model=model,
        ...     kv_pages=cache,
        ...     use_aot_forward=True,
        ...     min_input_pad=8,
        ...     max_model_len=8192,
        ...     max_num_reqs=32,
        ... )
        >>>
        >>> # Pre-compile for expected configurations
        >>> executor.compile(
        ...     num_tokens_paddings=[128, 256, 512, 1024, 2048],
        ...     num_reqs_max_model_len=16,
        ...     max_pages_per_req=128,
        ...     max_num_reqs=32,
        ...     metadata=cache.metadata,
        ... )
        >>>
        >>> # Execute steps during serving
        >>> results = executor.execute(
        ...     num_tokens=512,
        ...     device_state=state,
        ...     scheduled_full=scheduled,
        ...     req_num_tokens_full=req_tokens,
        ...     active_mask_full=active,
        ...     input_ids_buf=input_buf,
        ...     position_ids_buf=pos_buf,
        ...     padded_num_reqs=16,
        ... )
    """

    def __init__(
        self,
        model: EasyDeLBaseModule,
        use_aot_forward: bool = True,
        min_input_pad: int = 8,
        max_model_len: int = 2**13,
        max_num_reqs: int = 16,
        max_num_tokens: int | None = None,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig | None = None,
        verbose: bool = False,
        mpmd_scheduler: MpMdSchedulers | None = None,
        pipeline_plan: PipelineInferencePlan | None = None,
        pp_microbatch_count: int | str | None = "auto",
        pp_microbatch_size: int | str | None = "auto",
        full_hidden_state_max_tokens: int = 0,
        speculative_recurrent_state_tokens: int = 0,
    ):
        """Initialize the executor manager.

        Args:
            model: The EasyDeL model instance.
            use_aot_forward: Whether to use Ahead-of-Time (AOT) compilation for model
                execution. When True (default), functions are pre-compiled for better
                performance. When False, uses Just-In-Time (JIT) compilation with
                the graph definition passed as a static argument.
            min_input_pad: Minimum padding for inputs.
            max_model_len: Maximum model sequence length.
            max_num_reqs: Maximum number of requests.
            max_num_tokens: Maximum number of tokens for batching.
            metadata: Paged KV-cache config (ragged pages or unified attention).
            pp_microbatch_count: Expert PP decode wavefront policy. ``"auto"``
                keeps the built-in split, ``None`` or ``0`` disables
                microbatch wavefront execution, and a positive integer pins
                max microbatches per active window.
            pp_microbatch_size: Expert PP decode wavefront policy. ``"auto"``
                keeps the built-in split, ``None`` or ``0`` disables
                microbatch wavefront execution, and a positive integer pins
                rows per microbatch.
            full_hidden_state_max_tokens: Model-step buckets at or below this
                token count carry every hidden row. Used by runner-native
                speculative verification; standard decoding keeps compact
                sampled-row outputs.
            speculative_recurrent_state_tokens: Number of per-request recurrent
                candidate rows to allocate for speculative decode. Zero keeps
                the standard cache shape.
        """
        if metadata is None:
            raise ValueError("ExecutionManager requires a paged cache config `metadata`.")

        _attn_mech = model.config.get_text_config().attn_mechanism
        _cache_ver = metadata.version
        _max_tok = max_num_tokens if max_num_tokens is not None else max_model_len
        logger.info(f"initializing eSurge-ExecutionManager {_attn_mech} (cache={_cache_ver}, max_tokens={_max_tok})")
        self.model = model
        self.mesh = model.mesh

        self.use_aot_forward = use_aot_forward
        self.full_hidden_state_max_tokens = max(0, int(full_hidden_state_max_tokens))
        self.mpmd_scheduler = mpmd_scheduler
        self.pipeline_plan = pipeline_plan
        self.pp_microbatch_count = pp_microbatch_count
        self.pp_microbatch_size = pp_microbatch_size
        self.speculative_recurrent_state_tokens = max(0, int(speculative_recurrent_state_tokens))
        self.enable_spec_recurrent_commit_metadata = self.speculative_recurrent_state_tokens > 0
        # Slot-indexed state families (compressed-window caches) receive the
        # per-row physical state-slot ids in the batch metadata every step.
        self._slot_indexed_state = getattr(model, "esurge_cache_family", None) == "compressed_window"
        self.min_input_pad = min_input_pad
        self.max_model_len = max_model_len
        self.max_num_reqs = max_num_reqs
        if (
            bool(verbose)
            and self.pipeline_plan is not None
            and self.pipeline_plan.is_enabled
            and int(self.max_num_reqs) < int(self.pipeline_plan.mpmd_dim) * int(self.min_input_pad)
        ):
            logger.warning(
                "Forced PP inference is underfilled for decode wavefronts: pp=%d min_input_pad=%d "
                "requires at least %d active rows to fill the pipeline without shrinking buckets, "
                "but max_num_seqs=%d. Keeping PP enabled as requested; single-request/token latency "
                "will mostly measure serialized one-stage-per-chip execution.",
                int(self.pipeline_plan.mpmd_dim),
                int(self.min_input_pad),
                int(self.pipeline_plan.mpmd_dim) * int(self.min_input_pad),
                int(self.max_num_reqs),
            )
        self.max_num_tokens = max_num_tokens if max_num_tokens is not None else max_model_len
        self.metadata = metadata
        self._metadata_version = metadata.version
        self._use_slot_mapping = metadata.version == "v2"
        self._use_request_distribution = not self._use_slot_mapping

        text_config = model.config.get_text_config()
        kv_quant_cfg = text_config.kv_cache_quantization_config
        _is_turboquant = isinstance(kv_quant_cfg, TurboQuantConfig)
        quantizer = model._quant_class(quantization_config=None if _is_turboquant else kv_quant_cfg)
        self._cache_quantizer = quantizer
        self._cache_masking_details = getattr(text_config, "get_mask_details", lambda: None)()

        # Prefer HybridCache (per-operation cache views) as the universal container.
        # Keep paged-cache parameters consistent with the scheduler config.
        self.kv_pages = self._init_operations_cache_with_retry(
            quantizer=self._cache_quantizer,
            masking_details=self._cache_masking_details,
        )

        self.graphdef, self.graphstate, self.graphother = model.split_module()

        self.log_it = logger.info if verbose else logger.debug
        self._verbose = verbose

        self._empty_sharding = replicated_named_sharding(model.mesh)
        self._input_sharding = self._empty_sharding
        # Greedy-fastpath dispatch caches: in steady decode the valid mask
        # and total-token scalar are identical step over step, so the
        # host->device transfers they'd otherwise enqueue are pure waste.
        self._greedy_mask_cache: tuple[bytes, jax.Array] | None = None
        self._greedy_total_tokens_cache: dict[int, jax.Array] = {}
        if int(getattr(metadata, "data_parallel_size", 1) or 1) > 1:
            resolver = getattr(metadata, "runtime_sharding_resolver", None) or text_config.runtime_sharding_resolver
            self._input_sharding = NamedSharding(
                model.mesh,
                PartitionSpec(resolve_attention_data_parallel_axis(resolver)),
            )

        self._sampler_sharding = self._empty_sharding
        self.rng_key = jax.device_put(jax.random.PRNGKey(0), self._sampler_sharding)

        self._debug_baselines = {}

        self._batch_preparer = BatchMetadataPreparer(
            metadata=self.metadata,
            empty_sharding=self._empty_sharding,
            max_num_tokens=self.max_num_tokens,
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            min_input_pad=self.min_input_pad,
            enable_spec_recurrent_commit=self.enable_spec_recurrent_commit_metadata,
            input_sharding=self._input_sharding,
            slot_indexed_state=self._slot_indexed_state,
        )
        self._model_executor = ModelStepExecutor(
            model=self.model,
            mesh=self.mesh,
            metadata=self.metadata,
            kv_pages_template=self.kv_pages,
            graphstate_template=self.graphstate,
            graphother_template=self.graphother,
            max_num_reqs=self.max_num_reqs,
            graphdef=self.graphdef,
            empty_sharding=self._empty_sharding,
            use_aot_forward=self.use_aot_forward,
            mpmd_scheduler=self.mpmd_scheduler,
            pipeline_plan=self.pipeline_plan,
            full_hidden_state_max_tokens=self.full_hidden_state_max_tokens,
            enable_spec_recurrent_commit=self.enable_spec_recurrent_commit_metadata,
            input_sharding=self._input_sharding,
            slot_indexed_state=self._slot_indexed_state,
        )
        self._sampler_vocab_size = int(self.model.config.get_text_config().vocab_size)
        self._sampler_executor = SamplerExecutor(
            model=self.model,
            max_model_len=self.max_model_len,
            empty_sharding=self._sampler_sharding,
            use_aot_forward=self.use_aot_forward,
        )
        self._sampler_min_input_pad = 1
        # Sampler runtime (phase 11.1): CPU scratch buffers, penalty-count state,
        # rebuild/scatter jits, compaction, and the sample_tokens wrapper live on
        # SamplerRuntime. ExecutionManager keeps thin delegators plus read-only
        # back-reference properties (below) for names read inline in execute().
        self.sampler_runtime = SamplerRuntime(
            sampler_executor=self._sampler_executor,
            vocab_size=self._sampler_vocab_size,
            max_num_reqs=self.max_num_reqs,
            min_input_pad=self._sampler_min_input_pad,
            sampler_sharding=self._sampler_sharding,
            empty_sharding=self._empty_sharding,
        )
        self._req_num_tokens_placeholder = jnp.zeros(
            (self.max_num_reqs,),
            dtype=jnp.int32,
            out_sharding=self._empty_sharding,
        )
        self._model_num_tokens_paddings: list[int] = []
        # PP decode microbatching (phase 11.2): wavefront scratch/caches and the
        # try-execute path live on PipelineMicrobatchExecutor; only built when
        # the pipeline plan is enabled.
        self._pp_microbatch: PipelineMicrobatchExecutor | None = None
        if self.pipeline_plan is not None and self.pipeline_plan.is_enabled:
            self._pp_microbatch = PipelineMicrobatchExecutor(self)

    def _init_operations_cache_with_retry(self, *, quantizer: tp.Any, masking_details: tp.Any) -> tp.Any:
        """Allocate the model's operations cache, shrinking pages on PP HBM-OOM.

        Wraps :meth:`EasyDeLBaseModule.init_operations_cache` with a bounded
        retry loop that only fires when (a) the active pipeline plan is
        enabled and (b) the failure looks like an XLA OOM
        (``RESOURCE_EXHAUSTED`` / ``RuntimeBufferAllocationFailure``). On
        each retry the page count carried on ``self.metadata`` is multiplied
        by 0.78 (subject to ``num_pages >= 1``) and the allocation is tried
        again. Non-PP allocations (and non-OOM exceptions) re-raise after
        the first attempt because their page count is already validated by
        :class:`PipelineInferencePlan`.

        Args:
            quantizer: KV-cache quantizer instance built from the model's
                ``kv_cache_quantization_config`` (``None`` for unquantized).
            masking_details: Optional mask metadata returned by the model's
                ``get_mask_details``; passed through to
                :meth:`init_operations_cache`.

        Returns:
            The freshly-allocated KV pages cache returned by
            :meth:`init_operations_cache`.

        Raises:
            Whatever the underlying allocation raises on the final attempt
            (or immediately for non-OOM / non-PP failures).
            RuntimeError: If the retry loop terminates without success or
                a raise (defensive — should never happen).
        """

        def _allocate() -> tp.Any:
            """Single attempt at building the operations cache via the model API."""
            cache_batch_size = int(self.max_num_reqs)
            if self.speculative_recurrent_state_tokens > 0:
                cache_batch_size += cache_batch_size * int(self.speculative_recurrent_state_tokens)
            return self.model.init_operations_cache(
                batch_size=cache_batch_size,
                max_length=int(self.max_model_len),
                page_size=int(getattr(self.metadata, "page_size", 128)),
                hbm_utilization=float(getattr(self.metadata, "hbm_utilization", 0.9)),
                dtype=getattr(self.metadata, "kvdtype", None),
                quantizer=quantizer,
                masking_details=masking_details,
                ragged_config=self.metadata if isinstance(self.metadata, RaggedPagesCacheConfig) else None,
                unified_config=self.metadata if isinstance(self.metadata, UnifiedAttentionCacheConfig) else None,
                pipeline_plan=self.pipeline_plan,
            )

        max_attempts = 10 if getattr(self.pipeline_plan, "is_enabled", False) else 1
        for attempt in range(max_attempts):
            try:
                return _allocate()
            except Exception as exc:
                msg = str(exc)
                oom = "RESOURCE_EXHAUSTED" in msg or "RuntimeBufferAllocationFailure" in msg
                current_pages = metadata_num_pages(self.metadata)
                if not oom or current_pages is None or attempt + 1 >= max_attempts:
                    raise
                next_pages = max(1, int(current_pages * 0.78))
                if next_pages >= current_pages:
                    raise
                logger.warning(
                    "PP cache allocation failed with num_pages=%s; retrying with num_pages=%s (%s/%s).",
                    current_pages,
                    next_pages,
                    attempt + 1,
                    max_attempts,
                )
                set_metadata_num_pages(self.metadata, next_pages)

        raise RuntimeError("unreachable: cache allocation retry loop exhausted")

    def clear_cache(self) -> None:
        """Clear all cached compiled functions.

        Removes all cached compiled model and sampler functions, forcing
        recompilation on subsequent calls. Also clears debug hash baselines.

        Note:
            This is called automatically at the start of compile() when using
            AOT mode. May be called manually when model weights change
            significantly or when freeing memory is needed.
        """
        self._model_executor.clear_cache()
        self._sampler_executor.clear_cache()
        self._debug_baselines.clear()
        if self._pp_microbatch is not None:
            self._pp_microbatch.clear_cache()

    def invalidate_sampler_penalty_state(
        self,
        token_ids_cpu: numpy.ndarray | None = None,
        seq_lens_cpu: numpy.ndarray | None = None,
    ) -> None:
        """Delegate to :meth:`SamplerRuntime.invalidate_sampler_penalty_state`.

        Kept as an ExecutionManager entrypoint because the runner calls it on
        every SequenceBuffer row reordering (swap_rows / condense / layout
        version change).
        """
        self.sampler_runtime.invalidate_sampler_penalty_state(token_ids_cpu, seq_lens_cpu)

    def _ensure_sampler_penalty_state(self) -> None:
        """Delegate to :meth:`SamplerRuntime._ensure_sampler_penalty_state`."""
        self.sampler_runtime._ensure_sampler_penalty_state()

    def _prepare_compact_sampler_window(self, **kwargs) -> tuple[int, int, int]:
        """Delegate to :meth:`SamplerRuntime._prepare_compact_sampler_window`."""
        return self.sampler_runtime._prepare_compact_sampler_window(**kwargs)

    # CPU-window helpers moved to executors/sampler_executor.py (phase 11.1);
    # kept as staticmethod aliases for the call sites below and external users.
    _cpu_vector_size = staticmethod(_cpu_vector_size)
    _padded_cpu_window = staticmethod(_padded_cpu_window)

    # SamplerRuntime back-references (phase 11.1)
    # execute() and the PP microbatch path read the sampler scratch inline at
    # many sites. The numpy buffers are mutated in place by the runtime and
    # never rebound, so read-only property views preserve behavior exactly.
    # `_sampler_token_counts` is the only rebound leaf and gets a setter; the
    # single-writer rule (scheduler thread only) is unchanged.

    @property
    def _sampler_gather_positions_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_gather_positions_cpu

    @property
    def _sampler_sampling_seeds_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_sampling_seeds_cpu

    @property
    def _sampler_scatter_positions_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_scatter_positions_cpu

    @property
    def _sampler_window_row_indices_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_window_row_indices_cpu

    @property
    def _sampler_scheduled_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_scheduled_cpu

    @property
    def _sampler_seq_lens_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_seq_lens_cpu

    @property
    def _sampler_active_mask_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_active_mask_cpu

    @property
    def _sampler_temperature_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_temperature_cpu

    @property
    def _sampler_top_p_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_top_p_cpu

    @property
    def _sampler_top_k_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_top_k_cpu

    @property
    def _sampler_min_p_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_min_p_cpu

    @property
    def _sampler_frequency_penalties_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_frequency_penalties_cpu

    @property
    def _sampler_presence_penalties_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_presence_penalties_cpu

    @property
    def _sampler_repetition_penalties_cpu(self) -> numpy.ndarray:
        return self.sampler_runtime._sampler_repetition_penalties_cpu

    @property
    def _sampler_prefix_cpu_by_reqs(self) -> dict[int, numpy.ndarray]:
        return self.sampler_runtime._sampler_prefix_cpu_by_reqs

    @property
    def _sampler_zero_token_counts(self) -> jax.Array:
        return self.sampler_runtime._sampler_zero_token_counts

    @property
    def _sampler_token_counts(self) -> jax.Array:
        return self.sampler_runtime._sampler_token_counts

    @_sampler_token_counts.setter
    def _sampler_token_counts(self, value: jax.Array) -> None:
        self.sampler_runtime._sampler_token_counts = value

    def _pipeline_model_logits_bucket(
        self,
        *,
        padded_num_reqs: int,
        sampler_padded_num_reqs: int,
    ) -> int:
        """Return the fused PP model-step request bucket for compact logits.

        Sampler compaction can use tiny request buckets because top-k/top-p only
        needs rows that can emit a token. The fused PP model step belongs to the
        model request-bucket family, so it must still honor ``min_input_pad``.
        Without this floor, a one-row sampler window can select dead fused PP
        model variants such as ``8 tokens / 1 req`` even when the runner was
        configured with ``min_input_pad=4``.
        """
        padded_num_reqs = int(padded_num_reqs)
        sampler_padded_num_reqs = int(sampler_padded_num_reqs)
        if padded_num_reqs <= 0:
            return 0
        min_model_bucket = min(max(1, int(self.min_input_pad)), padded_num_reqs)
        return min(max(sampler_padded_num_reqs, min_model_bucket), padded_num_reqs)

    def _should_use_fused_pipeline_model_step(self, *, num_tokens: int, padded_num_reqs: int) -> bool:
        """Return whether a bucket should use the fused PP model-step executable.

        The fused PP path is a decode-latency optimization: it keeps the
        pipeline-parallel backbone, sampled hidden-row gather, and final-stage
        LM head in one SpectraX plan. It is only a good fit for decode-shaped
        windows where the token bucket is no wider than the request bucket
        (usually one new token per active request). Prefill-shaped windows such
        as ``16 tokens / 4 reqs`` carry multiple prompt tokens per request; those
        should use the split true-PP backbone plus the final-stage LM-head
        executable to avoid compiling a large fused prefill+projection graph.
        """
        if not self.model.mesh.is_mpmd:
            return False
        if not self._model_executor.supports_pipeline_model_step:
            return False
        return int(num_tokens) <= int(padded_num_reqs)

    @staticmethod
    def _is_decode_only_window(
        *,
        scheduled_full_cpu: numpy.ndarray,
        active_mask_full_cpu: numpy.ndarray,
        padded_num_reqs: int,
    ) -> bool:
        """Return whether every active scheduled row is a one-token decode row.

        Padding can make a prompt chunk look like a decode bucket: one request
        with four scheduled prompt tokens is represented as ``4 tokens / 4
        padded requests``. The resident PP runtime is only valid for the true
        decode case, where every active scheduled row contributes exactly one
        token. This predicate looks at the unpadded scheduler payload rather
        than the compile bucket.
        """
        padded_num_reqs = int(padded_num_reqs)
        scheduled_window = ExecutionManager._padded_cpu_window(
            scheduled_full_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.int32,
            fill_value=0,
        )
        active_window = ExecutionManager._padded_cpu_window(
            active_mask_full_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.bool_,
            fill_value=False,
        )
        active_scheduled = scheduled_window[active_window & (scheduled_window > 0)]
        return bool(active_scheduled.size > 0 and numpy.all(active_scheduled == 1))

    @staticmethod
    def _window_emits_token(
        *,
        scheduled_full_cpu: numpy.ndarray,
        active_mask_full_cpu: numpy.ndarray,
        req_num_tokens_full_cpu: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        padded_num_reqs: int,
    ) -> bool:
        """Return whether this scheduler window can produce a sampled token.

        Chunked prefill schedules prompt tokens before a request reaches its
        prompt target length. Intermediate chunks only need the backbone/cache
        update; projecting vocabulary logits and sampling them is wasted work.
        This mirrors the sampler's device-side validity condition:
        ``num_computed + scheduled >= req_num_tokens``.
        """
        pnr = int(padded_num_reqs)
        scheduled_window = ExecutionManager._padded_cpu_window(
            scheduled_full_cpu,
            padded_num_reqs=pnr,
            dtype=numpy.int32,
            fill_value=0,
        )
        active_window = ExecutionManager._padded_cpu_window(
            active_mask_full_cpu,
            padded_num_reqs=pnr,
            dtype=numpy.bool_,
            fill_value=False,
        )
        available_count = min(
            pnr,
            ExecutionManager._cpu_vector_size(scheduled_full_cpu),
            ExecutionManager._cpu_vector_size(active_mask_full_cpu),
            ExecutionManager._cpu_vector_size(req_num_tokens_full_cpu),
            ExecutionManager._cpu_vector_size(num_computed_tokens_cpu),
        )
        if available_count < pnr:
            active_window[available_count:] = False
        active_scheduled = active_window & (scheduled_window > 0)
        if not bool(numpy.any(active_scheduled)):
            return False
        seq_lens_now = (
            ExecutionManager._padded_cpu_window(
                num_computed_tokens_cpu,
                padded_num_reqs=pnr,
                dtype=numpy.int32,
                fill_value=0,
            )
            + scheduled_window
        )
        target_lens = ExecutionManager._padded_cpu_window(
            req_num_tokens_full_cpu,
            padded_num_reqs=pnr,
            dtype=numpy.int32,
            fill_value=numpy.iinfo(numpy.int32).max,
        )
        return bool(numpy.any(seq_lens_now[active_scheduled] >= target_lens[active_scheduled]))

    def has_compiled_variants(self) -> bool:
        """Check whether both model and sampler executors have compiled variants.

        Returns:
            True if both the model executor and the sampler executor have at
            least one cached compiled function, False otherwise.
        """
        return bool(self._model_executor.cache_keys()) and bool(self._sampler_executor.cache_keys())

    def _has_backbone_variant(self, num_tokens: int, *, use_pipeline_runtime: bool = True) -> bool:
        """Compatibility wrapper for checking a compiled backbone bucket."""
        try:
            return bool(
                self._model_executor.has_backbone(
                    int(num_tokens),
                    use_pipeline_runtime=bool(use_pipeline_runtime),
                )
            )
        except TypeError:
            return bool(self._model_executor.has_backbone(int(num_tokens)))

    def update_graphs(
        self,
        model: EasyDeLBaseModule | None = None,
        *,
        graphdef=None,
        graphstate=None,
        graphother=None,
    ) -> None:
        """Update the graph components (weights) used by the fused executor.

        Args:
            model: Optional EasyDeL module to source new graph parts from. When
                provided, graphdef/graphstate/graphother are pulled from this
                model unless explicitly overridden via the keyword arguments.
            graphdef: Optional graph definition replacement.
            graphstate: Optional graph state replacement (typically the weights).
            graphother: Optional auxiliary graph data replacement.

        Raises:
            ValueError: If neither a model nor explicit graph components are
                provided.
        """
        if model is not None:
            self.model = model
            self._model_executor.model = model
            self._sampler_executor.model = model
            if graphdef is None or graphstate is None or graphother is None:
                new_graphdef, new_graphstate, new_graphother = model.split_module()
                graphdef = new_graphdef if graphdef is None else graphdef
                graphstate = new_graphstate if graphstate is None else graphstate
                graphother = new_graphother if graphother is None else graphother

        if graphdef is None and graphstate is None and graphother is None:
            raise ValueError("No graph components supplied for update")

        if graphdef is not None:
            self.graphdef = graphdef
            self._model_executor.graphdef = graphdef

        if graphstate is not None:
            template_graphstate = self.graphstate if self.graphstate is not None else graphstate
            shardings = spx.extract_sharding_structure(template_graphstate, mesh=self.mesh)
            self.graphstate = jax.tree_util.tree_map(
                lambda x, s: jax.device_put(x, s) if hasattr(x, "dtype") else x,
                graphstate,
                shardings,
            )

        if graphother is not None:
            template_graphother = self.graphother if self.graphother is not None else graphother
            shardings = spx.extract_sharding_structure(template_graphother, mesh=self.mesh)
            self.graphother = jax.tree_util.tree_map(
                lambda x, s: jax.device_put(x, s) if hasattr(x, "dtype") else x,
                graphother,
                shardings,
            )

        if (
            self.mesh.is_mpmd
            and self.graphdef is not None
            and self.graphstate is not None
            and self.graphother is not None
        ):
            self._model_executor.refresh_lm_head_state(
                graphdef=self.graphdef,
                graphstate=self.graphstate,
                graphother=self.graphother,
            )
        if self.graphstate is not None and self.graphother is not None:
            self._model_executor.set_runtime_graph_args(self.graphstate, self.graphother)

        self._debug_baselines.clear()

    def execute(
        self,
        num_tokens: int,
        scheduled_full_cpu: numpy.ndarray,  # CPU array
        req_num_tokens_full_cpu: numpy.ndarray,
        active_mask_full_cpu: numpy.ndarray,  # CPU array
        window_row_indices_cpu: numpy.ndarray,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        padded_num_reqs: int,
        token_ids_cpu: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        temperature_cpu: numpy.ndarray,
        top_p_cpu: numpy.ndarray,
        top_k_cpu: numpy.ndarray,
        min_p_cpu: numpy.ndarray,
        frequency_penalties_cpu: numpy.ndarray,
        presence_penalties_cpu: numpy.ndarray,
        repetition_penalties_cpu: numpy.ndarray,
        page_table_cpu: numpy.ndarray,
        recurrent_slot_indices_cpu: numpy.ndarray | None = None,
        page_table_version: int | None = None,
        spec_recurrent_commit_cpu: numpy.ndarray | None = None,
        # VLM prefill helpers (optional)
        mrope_position_ids_cpu: numpy.ndarray | None = None,
        prefill_embeds_cpu: numpy.ndarray | None = None,
        prefill_embeds_mask_cpu: numpy.ndarray | None = None,
        # DeepStack-style visual injection (optional)
        visual_pos_masks_cpu: numpy.ndarray | None = None,
        deepstack_visual_embeds_cpu: list[numpy.ndarray] | None = None,
        # Vision-language model data (optional)
        pixel_values: numpy.ndarray | None = None,
        image_grid_thw: numpy.ndarray | None = None,
        pixel_values_videos: numpy.ndarray | None = None,
        video_grid_thw: numpy.ndarray | None = None,
        device_token_handoff: DeviceInputTokenHandoff | None = None,
        wait_for_outputs: bool = True,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array | None,
        dict[str, float | int],
    ]:
        """Execute a single fused inference step.

        Runs a pre-compiled function that combines input preparation, model
        forward pass, and token sampling in a single device dispatch.

        Args:
            num_tokens: Total tokens to process across all requests in this step.
                Must match a value from num_tokens_paddings used during compile().
            scheduled_full: Number of tokens scheduled per request [max_num_reqs].
                Determines how many tokens from each request enter this step.
            req_num_tokens_full_cpu: Target token count per request [max_num_reqs].
                Packed into sampler metadata to determine when requests have
                generated enough tokens without an extra per-step device_put.
            active_mask_full: Boolean mask for active requests [max_num_reqs].
                Inactive requests are skipped during processing.
            window_row_indices_cpu: Per-row global request indices for the
                current window. Used to map compact sampler rows back to the
                full-table layout.
            recurrent_slot_indices_cpu: Optional physical recurrent-state slots
                aligned with compact window rows. Rank-major DP uses this to
                keep GDN/Mamba state independent of movable sampler rows.
            input_ids_buf: Contiguous token ID buffer [max_num_tokens]. Flattened
                across requests for efficient batch processing.
            position_ids_buf: Contiguous position ID buffer [max_num_tokens].
                Parallel to input_ids_buf with position indices.
            padded_num_reqs: Bucketed request count for compilation lookup. Must
                be a power of 2 (or min_input_pad) matching a compiled variant.
            token_ids_cpu: Host-side ``[max_num_reqs, max_model_len]`` token id
                buffer used to fill the packed input slice.
            num_computed_tokens_cpu: Tokens already prefilled per request.
            temperature_cpu: Per-request sampling temperatures.
            top_p_cpu: Per-request top-p sampling values.
            top_k_cpu: Per-request top-k sampling values.
            min_p_cpu: Per-request min-p sampling thresholds.
            frequency_penalties_cpu: Per-request frequency penalty values.
            presence_penalties_cpu: Per-request presence penalty values.
            repetition_penalties_cpu: Per-request repetition penalty values.
            page_table_cpu: Host-side KV-cache page-table block ids.
            page_table_version: Optional monotonic version for page-table cache
                reuse; bump on allocator/swap/condense changes.
            spec_recurrent_commit_cpu: Optional speculative recurrent commit
                metadata (active mask + real prefix length per request).
            mrope_position_ids_cpu: Optional ``[3, num_tokens]`` mRoPE positions.
            prefill_embeds_cpu: Optional prefill embedding overrides for VLMs.
            prefill_embeds_mask_cpu: Optional mask selecting prefill rows.
            visual_pos_masks_cpu: Optional visual position masks for DeepStack.
            deepstack_visual_embeds_cpu: Optional DeepStack visual embeddings.
            pixel_values: Optional raw image pixel values for VLMs.
            image_grid_thw: Optional image grid shape ``(T, H, W)``.
            pixel_values_videos: Optional raw video pixel values for VLMs.
            video_grid_thw: Optional video grid shape ``(T, H, W)``.
            device_token_handoff: Optional device-resident sampled-token handoff
                from the previous PP step (avoids host token materialization).
            wait_for_outputs: If ``True``, block on the device-side output
                arrays before returning so the timing metrics reflect full
                completion; otherwise the returned arrays may still be running.

        Returns:
            Tuple of 7 elements:
                - out_tokens_full: Generated tokens [max_num_reqs], -1 for invalid.
                - valid_mask_full: Boolean mask for valid generations [max_num_reqs].
                - input_ids_buf: Updated input buffer (may contain new tokens).
                - position_ids_buf: Updated position buffer.
                - hidden_states: Last layer hidden states [num_tokens, hidden_dim].
                - logits: Output logits [padded_num_reqs, vocab_size], or ``None``
                  for prefill-only steps that do not sample.
                - metrics: Execution timing + bucket info.

        Raises:
            KeyError: If no compiled function exists for (num_tokens, padded_num_reqs).
                This indicates the configuration wasn't included in compile() call.

        Note:
            The KV cache (self.kv_pages) and random key (self.rng_key) are updated
            in-place on self after dispatch. When ``wait_for_outputs`` is False,
            the returned arrays may still be executing on device and the timing
            metrics reflect host dispatch time rather than full completion time.

        Example:
            >>> results = executor.execute(
            ...     num_tokens=256,
            ...     device_state=state,
            ...     scheduled_full=jnp.array([4, 8, 2, ...]),
            ...     req_num_tokens_full=jnp.array([512, 256, 128, ...]),
            ...     active_mask_full=jnp.array([True, True, False, ...]),
            ...     input_ids_buf=input_buf,
            ...     position_ids_buf=pos_buf,
            ...     padded_num_reqs=16,
            ... )
            >>> new_state, tokens, valid, *rest = results
        """
        if self._verbose and num_tokens > 0:
            logger.info(
                "[esurge-prof-exec] enter tok=%d reqs=%d wait=%s",
                int(num_tokens),
                int(padded_num_reqs),
                bool(wait_for_outputs),
            )
        microbatch_result = None
        if self._pp_microbatch is not None:
            microbatch_result = self._pp_microbatch.try_execute(
                num_tokens=num_tokens,
                scheduled_full_cpu=scheduled_full_cpu,
                active_mask_full_cpu=active_mask_full_cpu,
                window_row_indices_cpu=window_row_indices_cpu,
                input_ids_buf=input_ids_buf,
                position_ids_buf=position_ids_buf,
                padded_num_reqs=padded_num_reqs,
                req_num_tokens_full_cpu=req_num_tokens_full_cpu,
                token_ids_cpu=token_ids_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                temperature_cpu=temperature_cpu,
                top_p_cpu=top_p_cpu,
                top_k_cpu=top_k_cpu,
                min_p_cpu=min_p_cpu,
                frequency_penalties_cpu=frequency_penalties_cpu,
                presence_penalties_cpu=presence_penalties_cpu,
                repetition_penalties_cpu=repetition_penalties_cpu,
                page_table_cpu=page_table_cpu,
                mrope_position_ids_cpu=mrope_position_ids_cpu,
                prefill_embeds_cpu=prefill_embeds_cpu,
                prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
                visual_pos_masks_cpu=visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                device_token_handoff=device_token_handoff,
                wait_for_outputs=wait_for_outputs,
            )
        if microbatch_result is not None:
            if self._verbose and num_tokens > 0:
                logger.info("[esurge-prof-exec] microbatch path returned tok=%d", int(num_tokens))
            return microbatch_result

        start_prep = time.time()
        prep_phase_start = start_prep
        (
            batch_metadata,
            input_ids_buf,
            position_ids_buf,
            scheduled_full,
            active_mask_full,
        ) = self._batch_preparer.prepare_batch_metadata(
            num_tokens_static=num_tokens,
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            token_ids_cpu=token_ids_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            temperature_cpu=temperature_cpu,
            top_p_cpu=top_p_cpu,
            top_k_cpu=top_k_cpu,
            min_p_cpu=min_p_cpu,
            frequency_penalties_cpu=frequency_penalties_cpu,
            presence_penalties_cpu=presence_penalties_cpu,
            repetition_penalties_cpu=repetition_penalties_cpu,
            page_table_cpu=page_table_cpu,
            page_table_version=page_table_version,
            window_row_indices_cpu=window_row_indices_cpu,
            recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
            spec_recurrent_commit_cpu=spec_recurrent_commit_cpu,
            padded_num_reqs_in=padded_num_reqs,
            mrope_position_ids_cpu=mrope_position_ids_cpu,
            prefill_embeds_cpu=prefill_embeds_cpu,
            prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
            visual_pos_masks_cpu=visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
            # Vision-language model data
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        prep_batch_metadata_took = time.time() - prep_phase_start
        if self._verbose and num_tokens > 0:
            logger.info("[esurge-prof-exec] prepared metadata tok=%d", int(num_tokens))
        prep_handoff_took = 0.0
        if device_token_handoff is not None:
            prep_phase_start = time.time()
            batch_metadata = self._apply_device_token_handoff(batch_metadata, device_token_handoff)
            input_ids_buf = batch_metadata.input_ids_buf
            prep_handoff_took = time.time() - prep_phase_start

        window_emits_token = self._window_emits_token(
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            req_num_tokens_full_cpu=req_num_tokens_full_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            padded_num_reqs=padded_num_reqs,
        )
        use_pipeline_runtime_for_window = bool(
            self.model.mesh.is_mpmd
            and self._model_executor.pipeline_runtime is not None
            and self._is_decode_only_window(
                scheduled_full_cpu=scheduled_full_cpu,
                active_mask_full_cpu=active_mask_full_cpu,
                padded_num_reqs=padded_num_reqs,
            )
        )
        if self.model.mesh.is_mpmd and not window_emits_token:
            prep_ensure_variants_took = 0.0
            if not self._has_backbone_variant(
                int(num_tokens),
                use_pipeline_runtime=bool(use_pipeline_runtime_for_window),
            ):
                raise RuntimeError(
                    "Missing precompiled MPMD prefill backbone bucket "
                    f"num_tokens={int(num_tokens)} use_pipeline_runtime={bool(use_pipeline_runtime_for_window)}. "
                    "Runtime compilation is disabled; include this bucket in compile()."
                )
            if self._verbose and num_tokens > 0:
                logger.info(
                    "[esurge-prof-exec] prefill-only backbone ready tok=%d pp_runtime=%s",
                    int(num_tokens),
                    bool(use_pipeline_runtime_for_window),
                )

            prep_phase_start = time.time()
            inputs = StepFunctionInputs(
                kv_pages=self.kv_pages,
                scheduled_full=scheduled_full,
                req_num_tokens_full=self._req_num_tokens_placeholder,
                active_mask_full=active_mask_full,
                rng_key=self.rng_key,
                batch_metadata=batch_metadata,
            )
            if self._verbose and SYNC_INPUTS_FOR_TIMING:
                inputs = jax.block_until_ready(inputs)
            prep_pack_inputs_took = time.time() - prep_phase_start
            prep_took = time.time() - start_prep

            start_exec = time.time()
            if self._verbose and num_tokens > 0:
                logger.info("[esurge-prof-exec] launching prefill-only backbone tok=%d", int(num_tokens))
            backbone_fn = self._model_executor.get_backbone(
                num_tokens=num_tokens,
                use_pipeline_runtime=use_pipeline_runtime_for_window,
            )
            with forward_autotune_only():
                backbone_outputs = backbone_fn(self.graphstate, self.graphother, inputs.kv_pages, inputs.batch_metadata)
            self.kv_pages = backbone_outputs.kv_pages
            if wait_for_outputs:
                if self._verbose and num_tokens > 0:
                    logger.info("[esurge-prof-exec] waiting prefill-only backbone tok=%d", int(num_tokens))
                jax.block_until_ready(backbone_outputs.hidden_states)
            exec_took = time.time() - start_exec
            sample_took = 0.0
            execute_total_took = time.time() - start_prep
            execute_overhead_took = max(0.0, float(execute_total_took - (prep_took + exec_took + sample_took)))
            out_tokens_full = jax.device_put(
                numpy.full((int(padded_num_reqs),), -1, dtype=numpy.int32),
                self._empty_sharding,
            )
            valid_mask_full = jax.device_put(
                numpy.zeros((int(padded_num_reqs),), dtype=numpy.bool_),
                self._empty_sharding,
            )
            metrics = {
                "exec_time": exec_took,
                "sample_time": sample_took,
                "prep_time": prep_took,
                "prep_batch_metadata_time": prep_batch_metadata_took,
                "prep_handoff_time": prep_handoff_took,
                "prep_sampler_window_time": 0.0,
                "prep_ensure_variants_time": prep_ensure_variants_took,
                "prep_pack_inputs_time": prep_pack_inputs_took,
                "execute_overhead_time": execute_overhead_took,
                "buckets_processed": int(batch_metadata.input_ids_buf.shape[-1]),
                "token_bucket": int(num_tokens),
                "padded_num_reqs": int(padded_num_reqs),
                "sampler_padded_num_reqs": 0,
                "sampler_num_reqs": 0,
                "prefill_only": 1,
            }
            metrics.update(self._model_executor.last_pipeline_stats())
            metrics.update(self._batch_preparer.last_prep_stats)
            return (
                out_tokens_full,
                valid_mask_full,
                input_ids_buf,
                position_ids_buf,
                backbone_outputs.hidden_states,
                None,
                metrics,
            )

        prep_phase_start = time.time()
        sampler_num_reqs, sampler_padded_num_reqs, sampler_total_tokens = self._prepare_compact_sampler_window(
            padded_num_reqs=padded_num_reqs,
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            window_row_indices_cpu=window_row_indices_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            temperature_cpu=temperature_cpu,
            top_p_cpu=top_p_cpu,
            top_k_cpu=top_k_cpu,
            min_p_cpu=min_p_cpu,
            frequency_penalties_cpu=frequency_penalties_cpu,
            presence_penalties_cpu=presence_penalties_cpu,
            repetition_penalties_cpu=repetition_penalties_cpu,
        )
        prep_sampler_window_took = time.time() - prep_phase_start
        model_logits_padded_num_reqs = int(padded_num_reqs)
        sampler_prefix = self._sampler_prefix_cpu_by_reqs.get(int(sampler_padded_num_reqs))
        if sampler_prefix is None:
            sampler_prefix = numpy.arange(int(sampler_padded_num_reqs), dtype=numpy.int32)
            self._sampler_prefix_cpu_by_reqs[int(sampler_padded_num_reqs)] = sampler_prefix
        if self.model.mesh.is_mpmd and self._model_executor.supports_pipeline_model_step:
            if numpy.array_equal(
                self._sampler_gather_positions_cpu[: int(sampler_padded_num_reqs)],
                sampler_prefix,
            ):
                model_logits_padded_num_reqs = self._pipeline_model_logits_bucket(
                    padded_num_reqs=padded_num_reqs,
                    sampler_padded_num_reqs=sampler_padded_num_reqs,
                )
        prep_phase_start = time.time()
        self._require_precompiled_variants(
            num_tokens=num_tokens,
            padded_num_reqs=model_logits_padded_num_reqs,
            sampler_padded_num_reqs=sampler_padded_num_reqs,
            use_pipeline_runtime=use_pipeline_runtime_for_window,
        )
        prep_ensure_variants_took = time.time() - prep_phase_start
        if self._verbose and num_tokens > 0:
            logger.info(
                "[esurge-prof-exec] variants ready tok=%d model_reqs=%d sampler_reqs=%d pp_runtime=%s",
                int(num_tokens),
                int(model_logits_padded_num_reqs),
                int(sampler_padded_num_reqs),
                bool(use_pipeline_runtime_for_window),
            )

        prep_phase_start = time.time()
        inputs = StepFunctionInputs(
            kv_pages=self.kv_pages,
            scheduled_full=scheduled_full,
            req_num_tokens_full=self._req_num_tokens_placeholder,
            active_mask_full=active_mask_full,
            rng_key=self.rng_key,
            batch_metadata=batch_metadata,
        )
        # Syncing inputs here improves `prep_time` accuracy but adds a device
        # round-trip; keep it behind an explicit env flag.
        if self._verbose and SYNC_INPUTS_FOR_TIMING:
            inputs = jax.block_until_ready(inputs)
        prep_pack_inputs_took = time.time() - prep_phase_start
        prep_took = time.time() - start_prep
        if DEBUG_MODE:
            model_hash = _tree_hash((self.graphstate, self.graphother, inputs))
            model_hash_baseline = self._debug_baselines[f"{num_tokens}_hash_in_backbone"]
            _tree_hash_diff(model_hash_baseline, model_hash)

        need_penalties = bool(
            numpy.any(frequency_penalties_cpu != 0.0)
            or numpy.any(presence_penalties_cpu != 0.0)
            or numpy.any(repetition_penalties_cpu != 1.0)
        )
        sampler_active_cpu = self._sampler_active_mask_cpu[: int(sampler_padded_num_reqs)].astype(
            numpy.bool_,
            copy=False,
        )
        req_num_tokens_for_sampler = numpy.zeros((int(sampler_padded_num_reqs),), dtype=numpy.int32)
        if int(sampler_num_reqs) > 0:
            live_gather_positions = self._sampler_gather_positions_cpu[: int(sampler_num_reqs)]
            req_num_tokens_for_sampler[: int(sampler_num_reqs)] = req_num_tokens_full_cpu[live_gather_positions]
        valid_mask_cpu = (
            sampler_active_cpu
            & (self._sampler_scheduled_cpu[: int(sampler_padded_num_reqs)] > 0)
            & (self._sampler_seq_lens_cpu[: int(sampler_padded_num_reqs)] >= req_num_tokens_for_sampler)
        )
        use_greedy_argmax_fastpath = bool(
            not need_penalties
            # Under an MPMD/PP mesh the logits live on the final stage's
            # submesh, but the fastpath's helper tensors (valid mask, RNG
            # fold scalar) are replicated over the full mesh — a device-set
            # mismatch inside the jitted argmax. The standard sampler path
            # places its inputs on the stage submesh, so fall back to it for
            # pipeline meshes (greedy decode stays correct, just not fused).
            and not self._model_executor._uses_mpmd_mesh()
            and int(sampler_padded_num_reqs) == int(padded_num_reqs)
            and int(model_logits_padded_num_reqs) == int(padded_num_reqs)
            and numpy.array_equal(
                self._sampler_gather_positions_cpu[: int(sampler_padded_num_reqs)],
                sampler_prefix,
            )
            and numpy.array_equal(
                self._sampler_scatter_positions_cpu[: int(sampler_padded_num_reqs)],
                sampler_prefix,
            )
            and numpy.all(
                numpy.where(
                    sampler_active_cpu,
                    self._sampler_temperature_cpu[: int(sampler_padded_num_reqs)] <= 0.0,
                    True,
                )
            )
        )
        start_exec = time.time()
        if self._verbose and num_tokens > 0:
            logger.info("[esurge-prof-exec] launching model tok=%d", int(num_tokens))
        model_enqueue_start = time.time()
        model_outputs = self.execute_model(
            num_tokens=num_tokens,
            padded_num_reqs=model_logits_padded_num_reqs,
            inputs=inputs,
            use_pipeline_runtime=use_pipeline_runtime_for_window,
        )
        model_enqueue_took = time.time() - model_enqueue_start
        if self._verbose and num_tokens > 0:
            logger.info("[esurge-prof-exec] model launched tok=%d", int(num_tokens))

        sampler_inputs = (
            sampler_padded_num_reqs,
            sampler_num_reqs,
            sampler_total_tokens,
            self._sampler_gather_positions_cpu[:sampler_padded_num_reqs],
            self._sampler_sampling_seeds_cpu[:sampler_padded_num_reqs],
            self._sampler_scatter_positions_cpu[:sampler_padded_num_reqs],
            self._sampler_window_row_indices_cpu[:sampler_padded_num_reqs],
            self._sampler_scheduled_cpu[:sampler_padded_num_reqs],
            self._sampler_seq_lens_cpu[:sampler_padded_num_reqs],
            self._sampler_active_mask_cpu[:sampler_padded_num_reqs],
            self._sampler_temperature_cpu[:sampler_padded_num_reqs],
            self._sampler_top_p_cpu[:sampler_padded_num_reqs],
            self._sampler_top_k_cpu[:sampler_padded_num_reqs],
            self._sampler_min_p_cpu[:sampler_padded_num_reqs],
            self._sampler_frequency_penalties_cpu[:sampler_padded_num_reqs],
            self._sampler_presence_penalties_cpu[:sampler_padded_num_reqs],
            self._sampler_repetition_penalties_cpu[:sampler_padded_num_reqs],
        )

        if DEBUG_MODE:
            sampler_hash = _tree_hash(sampler_inputs)
            sampler_hash_baseline = self._debug_baselines[f"{num_tokens}_{sampler_padded_num_reqs}_hash_in_sampler"]
            _tree_hash_diff(sampler_hash_baseline, sampler_hash)

        # Enqueue sampling immediately (it will run after the forward pass),
        # then synchronize on logits to measure forward time without an extra
        # host-side dispatch gap between the two computations.
        greedy_argmax_took = 0.0
        if use_greedy_argmax_fastpath:
            sampler_enqueue_took = 0.0
            greedy_argmax_start = time.time()
            valid_mask_dev = self._greedy_valid_mask_device(valid_mask_cpu)
            rng_key_out, out_tokens_full, valid_mask_full = _greedy_argmax_tokens(
                model_outputs.logits,
                valid_mask_dev,
                self.rng_key,
                self._greedy_total_tokens_device(int(sampler_total_tokens)),
                padded_num_reqs=int(padded_num_reqs),
            )
            token_counts_out = self._sampler_token_counts
            greedy_argmax_took = time.time() - greedy_argmax_start
            if self._verbose and num_tokens > 0:
                logger.info("[esurge-prof-exec] greedy argmax launched tok=%d", int(num_tokens))
        else:
            sampler_enqueue_start = time.time()
            sampler_out = self.sample_tokens(
                num_tokens=num_tokens,
                padded_num_reqs=padded_num_reqs,
                sampler_padded_num_reqs=sampler_padded_num_reqs,
                sampler_num_reqs=sampler_num_reqs,
                sampler_total_tokens=sampler_total_tokens,
                req_num_tokens_full_cpu=req_num_tokens_full_cpu,
                logits=model_outputs.logits,
                rng_key=self.rng_key,
                gather_positions_cpu=self._sampler_gather_positions_cpu,
                sampling_seeds_cpu=self._sampler_sampling_seeds_cpu,
                scatter_positions_cpu=self._sampler_scatter_positions_cpu,
                compact_window_row_indices_cpu=self._sampler_window_row_indices_cpu,
                compact_scheduled_cpu=self._sampler_scheduled_cpu,
                compact_seq_lens_cpu=self._sampler_seq_lens_cpu,
                compact_active_mask_cpu=self._sampler_active_mask_cpu,
                compact_temperature_cpu=self._sampler_temperature_cpu,
                compact_top_p_cpu=self._sampler_top_p_cpu,
                compact_top_k_cpu=self._sampler_top_k_cpu,
                compact_min_p_cpu=self._sampler_min_p_cpu,
                compact_frequency_penalties_cpu=self._sampler_frequency_penalties_cpu,
                compact_presence_penalties_cpu=self._sampler_presence_penalties_cpu,
                compact_repetition_penalties_cpu=self._sampler_repetition_penalties_cpu,
                need_penalties=need_penalties,
            )
            sampler_enqueue_took = time.time() - sampler_enqueue_start
            if self._verbose and num_tokens > 0:
                logger.info("[esurge-prof-exec] sampler launched tok=%d", int(num_tokens))
            rng_key_out, out_tokens_full, valid_mask_full, token_counts_out = sampler_out
        self.rng_key = rng_key_out
        self._sampler_token_counts = token_counts_out
        logits_wait_took = 0.0
        if wait_for_outputs:
            if self._verbose and num_tokens > 0:
                logger.info("[esurge-prof-exec] waiting model logits tok=%d", int(num_tokens))
            logits_wait_start = time.time()
            jax.block_until_ready(model_outputs.logits)
            logits_wait_took = time.time() - logits_wait_start
            exec_took = time.time() - start_exec

            start_sample = time.time()
            if self._verbose and num_tokens > 0:
                logger.info("[esurge-prof-exec] waiting sampler tok=%d", int(num_tokens))
            jax.block_until_ready(out_tokens_full)
            sample_took = time.time() - start_sample
        else:
            exec_took = time.time() - start_exec
            sample_took = 0.0

        execute_total_took = time.time() - start_prep
        execute_overhead_took = execute_total_took - (prep_took + exec_took + sample_took)
        execute_overhead_took = max(0.0, float(execute_overhead_took))
        buckets_processed = batch_metadata.input_ids_buf.shape[-1]
        metrics = {
            "exec_time": exec_took,
            "model_enqueue_time": model_enqueue_took,
            "sampler_enqueue_time": sampler_enqueue_took,
            "logits_wait_time": logits_wait_took,
            "exec_enqueue_time": model_enqueue_took + sampler_enqueue_took,
            "exec_wait_time": logits_wait_took,
            "sample_time": sample_took,
            "sampler_wait_time": sample_took,
            "prep_time": prep_took,
            "prep_batch_metadata_time": prep_batch_metadata_took,
            "prep_handoff_time": prep_handoff_took,
            "prep_sampler_window_time": prep_sampler_window_took,
            "prep_ensure_variants_time": prep_ensure_variants_took,
            "prep_pack_inputs_time": prep_pack_inputs_took,
            "execute_overhead_time": execute_overhead_took,
            "buckets_processed": buckets_processed,
            "token_bucket": int(num_tokens),
            "padded_num_reqs": int(padded_num_reqs),
            "sampler_padded_num_reqs": int(sampler_padded_num_reqs),
            "sampler_num_reqs": int(sampler_num_reqs),
            "greedy_argmax_time": greedy_argmax_took,
            "greedy_argmax_fastpath": int(use_greedy_argmax_fastpath),
        }
        metrics.update(self._model_executor.last_pipeline_stats())
        metrics.update(self._batch_preparer.last_prep_stats)
        if self._verbose and num_tokens > 0:
            logger.info(
                "[esurge-prof-exec] tok=%d reqs=%d sampler_reqs=%d prep=%.3fms "
                "prep_batch=%.3fms handoff=%.3fms samplerwin=%.3fms ensure=%.3fms "
                "pack=%.3fms fwd=%.3fms sample=%.3fms overhead=%.3fms total=%.3fms",
                int(num_tokens),
                int(padded_num_reqs),
                int(sampler_padded_num_reqs),
                prep_took * 1e3,
                prep_batch_metadata_took * 1e3,
                prep_handoff_took * 1e3,
                prep_sampler_window_took * 1e3,
                prep_ensure_variants_took * 1e3,
                prep_pack_inputs_took * 1e3,
                exec_took * 1e3,
                sample_took * 1e3,
                execute_overhead_took * 1e3,
                execute_total_took * 1e3,
            )

        hidden_states = model_outputs.hidden_states
        logits = model_outputs.logits

        return (
            out_tokens_full,
            valid_mask_full,
            input_ids_buf,
            position_ids_buf,
            hidden_states,
            logits,
            metrics,
        )

    # Device-token-handoff application moved to BatchMetadataPreparer (phase
    # 11.3); kept as a static alias because execute(), the PP microbatch
    # executor, and tests call it as ExecutionManager._apply_device_token_handoff.
    _apply_device_token_handoff = staticmethod(BatchMetadataPreparer._apply_device_token_handoff)

    def _greedy_valid_mask_device(self, valid_mask_cpu) -> jax.Array:
        """Device copy of the greedy valid mask, cached by content.

        Steady-state decode presents the same mask every step; keying on the
        raw bytes skips the per-step ``device_put`` dispatch while staying
        correct the moment membership changes.
        """
        key = valid_mask_cpu.tobytes()
        cached = self._greedy_mask_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        device_mask = jax.device_put(valid_mask_cpu, self._empty_sharding)
        self._greedy_mask_cache = (key, device_mask)
        return device_mask

    def _greedy_total_tokens_device(self, total_tokens: int) -> jax.Array:
        """Device scalar for the sampled-token RNG fold, cached per value.

        Bounded by the token-bucket ladder, so the cache stays tiny.
        """
        cached = self._greedy_total_tokens_cache.get(total_tokens)
        if cached is None:
            cached = jax.device_put(jnp.asarray(total_tokens, dtype=jnp.int32), self._empty_sharding)
            self._greedy_total_tokens_cache[total_tokens] = cached
        return cached

    def execute_model(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        inputs: StepFunctionInputs,
        use_pipeline_runtime: bool = True,
    ) -> ModelStepOutputs:
        """Run the compiled model forward step and update self.kv_pages.

        Executes the pre-compiled model step function, computing hidden states
        and logits while updating the KV cache with new attention states.

        Args:
            num_tokens: Number of tokens for bucket selection.
            padded_num_reqs: Padded request count for bucket selection.
            inputs: Consolidated step function inputs containing kv_pages,
                batch_metadata, and other required tensors.
            use_pipeline_runtime: Select the resident PP-runtime backbone
                variant for true decode wavefronts, or the direct SpectraX
                variant for prompt prefill chunks.

        Returns:
            ModelStepOutputs containing updated kv_pages, hidden_states, and
            logits.

        Note:
            This method updates self.kv_pages in-place with the new cache state.
            The returned outputs.kv_pages is the same reference. The method does
            not block on completion, allowing the caller to pipeline work like
            enqueuing sampling before synchronizing.
        """
        model_fn = self._model_executor.get_compiled(
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            use_pipeline_runtime=use_pipeline_runtime,
        )
        # Do not block here: allow the caller to pipeline dependent work
        # (e.g. enqueue sampling) before synchronizing.
        graphstate_arg, graphother_arg = self._model_executor.runtime_graph_args()
        with forward_autotune_only():
            outputs = model_fn(graphstate_arg, graphother_arg, inputs.kv_pages, inputs.batch_metadata)
        self.kv_pages = outputs.kv_pages
        return outputs

    def shutdown(self) -> None:
        """Release resources owned by sub-executors.

        Forwards to :meth:`ModelStepExecutor.shutdown`, which joins the
        resident PP-stage worker threads when running in MPMD mode. The
        sampler executor has no background threads so is not touched here.
        Safe to call multiple times.
        """
        if hasattr(self, "_model_executor"):
            self._model_executor.shutdown()

    def clear_recurrent_slots(self, slot_indices: list[int]) -> None:
        """Zero recurrent/SSM state in freed slots before slot reuse.

        Recurrent layers (Mamba/SSM, RWKV, parallel hybrids) keep
        ``conv_state`` and ``recurrent_state`` per request slot. When a
        request finishes and its row is later reassigned to a new request,
        the new request must start from clean state — otherwise the new
        request inherits stale activations from the previous occupant.
        This method writes zeros into those tensors at the indicated
        rows for every recurrent or hybrid layer present, leaving
        attention layers untouched.

        Compressed-window caches (DeepSeek-V4) reset the freed rows across
        every layer's ring/compressor/indexer state via
        :meth:`~easydel.caching.CompressedWindowCache.reset_slots`.

        No-op when the cache is neither a :class:`HybridCache` nor a
        :class:`~easydel.caching.CompressedWindowCache` (purely attention
        models have no per-slot state to clear) or when no slots were freed.

        Args:
            slot_indices: Row indices in ``[0, max_num_reqs)`` whose
                recurrent state should be zeroed. Empty list short-circuits.
        """
        if not slot_indices:
            return
        cache = self.kv_pages
        from easydel.caching import CompressedWindowCache

        if isinstance(cache, CompressedWindowCache):
            self.kv_pages = cache.reset_slots(numpy.asarray(sorted({int(s) for s in slot_indices}), dtype=numpy.int32))
            return
        if not isinstance(cache, HybridCache):
            return

        changed = False
        new_views = list(cache.views)
        slot_indices_to_clear = list(slot_indices)
        candidate_count = max(0, int(getattr(self, "speculative_recurrent_state_tokens", 0) or 0))
        if candidate_count > 0:
            base_slots = int(self.max_num_reqs)
            for slot in slot_indices:
                if 0 <= int(slot) < base_slots:
                    # Prefix-major candidate layout: base + prefix * base_slots + slot.
                    slot_indices_to_clear.extend(base_slots + k * base_slots + int(slot) for k in range(candidate_count))
        # Collect every recurrent leaf first, then zero the masked rows of all
        # of them in ONE fused dispatch per distinct pool size (a python loop of
        # eager wheres cost ~100 host dispatches per slot-clear event).
        entries: list[tuple[int, typing.Any, jax.Array | None, jax.Array | None, int]] = []
        for idx, view in enumerate(new_views):
            rec = None
            if isinstance(view, RecurrentCacheView):
                rec = view
            elif isinstance(view, ParallelHybridCacheView) and view.recurrent is not None:
                rec = view.recurrent
            else:
                continue
            conv_arr = rec.conv_state
            rec_arr = rec.recurrent_state
            if conv_arr is not None:
                n_slots = int(typing.cast(jax.Array, conv_arr).shape[0])
            elif rec_arr is not None:
                n_slots = int(typing.cast(jax.Array, rec_arr).shape[0])
            else:
                continue
            entries.append((idx, rec, conv_arr, rec_arr, n_slots))

        by_pool: dict[int, list[jax.Array]] = {}
        for _idx, _rec, conv_arr, rec_arr, n_slots in entries:
            bucket = by_pool.setdefault(n_slots, [])
            if conv_arr is not None:
                bucket.append(conv_arr)
            if rec_arr is not None:
                bucket.append(rec_arr)
        cleared_by_pool: dict[int, list[jax.Array]] = {}
        for n_slots, arrays in by_pool.items():
            filtered_slots = sorted({int(s) for s in slot_indices_to_clear if 0 <= int(s) < n_slots})
            if not filtered_slots:
                continue
            keep_np = numpy.ones((n_slots,), dtype=bool)
            keep_np[numpy.asarray(filtered_slots, dtype=numpy.int64)] = False
            transform = _jit_zero_rows(tuple(array.sharding for array in arrays))
            cleared_by_pool[n_slots] = list(transform(arrays, jnp.asarray(keep_np)))

        cursors = dict.fromkeys(cleared_by_pool, 0)
        for idx, rec, conv_arr, rec_arr, n_slots in entries:
            cleared = cleared_by_pool.get(n_slots)
            if cleared is None:
                continue
            new_conv = conv_arr
            new_rec = rec_arr
            if conv_arr is not None:
                new_conv = cleared[cursors[n_slots]]
                cursors[n_slots] += 1
            if rec_arr is not None:
                new_rec = cleared[cursors[n_slots]]
                cursors[n_slots] += 1
            new_recurrent = rec.replace(conv_state=new_conv, recurrent_state=new_rec)
            view = new_views[idx]
            if isinstance(view, ParallelHybridCacheView):
                new_views[idx] = view.replace(
                    recurrent=new_recurrent,
                    conv_state=new_conv,
                    recurrent_state=new_rec,
                )
            else:
                new_views[idx] = new_recurrent
            changed = True

        def sidecar_clear_indices(n_slots):
            selected = sorted({int(s) for s in slot_indices_to_clear if 0 <= int(s) < n_slots})
            if not selected:
                return None
            keep = numpy.ones((n_slots,), dtype=bool)
            keep[numpy.asarray(selected, dtype=numpy.int32)] = False
            return numpy.arange(n_slots, dtype=numpy.int32), keep

        changed = _transform_recurrent_sidecars(new_views, sidecar_clear_indices) or changed
        if changed:
            self.kv_pages = HybridCache(views=new_views)

    def permute_recurrent_slots(self, perm: numpy.ndarray) -> None:
        """Gather recurrent/SSM state rows into a new physical row layout.

        On the plain-recurrent path (no per-request slot pool) the conv/GDR/SSM
        cache is indexed by the physical ``SequenceBuffer`` row a request
        occupies in the packed batch — the ragged conv/GDN kernels read and
        write ``state_indices = arange(num_slots)``. When ``condense`` or
        ``reorder_decode_first`` relocates a surviving request to a different
        row (e.g. after a mid-stream request finishes), its device-side
        ``conv_state``/``recurrent_state`` must move in lockstep or the request
        reads a freed/zeroed neighbour's state and emits garbage. This method
        applies that row permutation to every recurrent or hybrid layer.

        No-op unless the cache is a :class:`HybridCache`. Rank-major SPMD DP and
        slot-indexed compressed-window families never use this path — they pin
        each request to a stable pooled slot and thread ``recurrent_state_indices``
        to the kernel instead (see :meth:`clear_recurrent_slots`).

        Args:
            perm: Integer array of length ``max_num_reqs``. ``perm[t]`` is the
                source row whose recurrent state should occupy destination row
                ``t``, or ``-1`` when destination ``t`` must be zeroed (a freed
                row, or a newly admitted request whose state is repopulated by
                its own prefill). Any speculative-candidate rows follow their
                owning base row automatically.
        """
        cache = self.kv_pages
        if not isinstance(cache, HybridCache):
            return
        base_slots = int(self.max_num_reqs)
        perm = numpy.asarray(perm, dtype=numpy.int32).reshape(-1)
        if perm.shape[0] < base_slots:
            perm = numpy.concatenate([perm, numpy.full((base_slots - perm.shape[0],), -1, dtype=numpy.int32)])
        else:
            perm = perm[:base_slots]
        candidate_count = max(0, int(getattr(self, "speculative_recurrent_state_tokens", 0) or 0))

        changed = False
        new_views = list(cache.views)
        # Collect every recurrent leaf first, then apply the whole permutation
        # in ONE fused dispatch per distinct pool size (the previous per-view
        # python loop cost ~100 eager gather+where dispatches per row-move
        # event, which scales with request churn and thus with concurrency).
        entries: list[tuple[int, typing.Any, jax.Array | None, jax.Array | None, int]] = []
        for idx, view in enumerate(new_views):
            if isinstance(view, RecurrentCacheView):
                rec = view
            elif isinstance(view, ParallelHybridCacheView) and view.recurrent is not None:
                rec = view.recurrent
            else:
                continue
            conv_arr = rec.conv_state
            rec_arr = rec.recurrent_state
            if conv_arr is not None:
                n_slots = int(typing.cast(jax.Array, conv_arr).shape[0])
            elif rec_arr is not None:
                n_slots = int(typing.cast(jax.Array, rec_arr).shape[0])
            else:
                continue
            entries.append((idx, rec, conv_arr, rec_arr, n_slots))

        def _gather_keep(n_slots: int) -> tuple[numpy.ndarray, numpy.ndarray] | None:
            """Full-pool gather index + keep mask for one pool size (or None if identity)."""
            gather = numpy.arange(n_slots, dtype=numpy.int32)
            keep = numpy.ones((n_slots,), dtype=bool)
            upper = min(base_slots, n_slots)
            base = perm[:upper]
            gather[:upper] = numpy.where(base >= 0, base, 0)
            keep[:upper] = base >= 0
            if candidate_count > 0 and n_slots > base_slots:
                # Prefix-major candidate layout: base + prefix * base_slots + slot.
                # Candidate rows follow their owning base row.
                for k in range(candidate_count):
                    lo = base_slots + k * base_slots
                    hi = min(lo + base_slots, n_slots)
                    width = hi - lo
                    if width <= 0:
                        break
                    seg_base = perm[:width]
                    src = lo + numpy.where(seg_base >= 0, seg_base, 0)
                    valid = (seg_base >= 0) & (src < n_slots)
                    gather[lo:hi] = numpy.where(valid, src, 0)
                    keep[lo:hi] = valid
            if keep.all() and numpy.array_equal(gather, numpy.arange(n_slots, dtype=numpy.int32)):
                return None
            return gather, keep

        by_pool: dict[int, list[jax.Array]] = {}
        for _idx, _rec, conv_arr, rec_arr, n_slots in entries:
            bucket = by_pool.setdefault(n_slots, [])
            if conv_arr is not None:
                bucket.append(conv_arr)
            if rec_arr is not None:
                bucket.append(rec_arr)
        moved_by_pool: dict[int, list[jax.Array]] = {}
        for n_slots, arrays in by_pool.items():
            gk = _gather_keep(n_slots)
            if gk is None:
                continue
            gather, keep = gk
            transform = _jit_permute_rows(tuple(array.sharding for array in arrays))
            moved_by_pool[n_slots] = list(transform(arrays, jnp.asarray(gather, dtype=jnp.int32), jnp.asarray(keep)))

        cursors = dict.fromkeys(moved_by_pool, 0)
        for idx, rec, conv_arr, rec_arr, n_slots in entries:
            moved = moved_by_pool.get(n_slots)
            if moved is None:
                continue
            new_conv = conv_arr
            new_rec = rec_arr
            if conv_arr is not None:
                new_conv = moved[cursors[n_slots]]
                cursors[n_slots] += 1
            if rec_arr is not None:
                new_rec = moved[cursors[n_slots]]
                cursors[n_slots] += 1
            new_recurrent = rec.replace(conv_state=new_conv, recurrent_state=new_rec)
            view = new_views[idx]
            if isinstance(view, ParallelHybridCacheView):
                new_views[idx] = view.replace(
                    recurrent=new_recurrent,
                    conv_state=new_conv,
                    recurrent_state=new_rec,
                )
            else:
                new_views[idx] = new_recurrent
            changed = True

        changed = _transform_recurrent_sidecars(new_views, _gather_keep) or changed
        if changed:
            self.kv_pages = HybridCache(views=new_views)

    def sample_tokens(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        **kwargs,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Delegate to :meth:`SamplerRuntime.sample_tokens`.

        The sampler step wrapper (compiled-variant lookup, packed metadata
        assembly, compact gather/scatter) moved to
        :class:`~easydel.inference.esurge.runners.executors.sampler_executor.SamplerRuntime`
        in phase 11.1; see it for the full argument documentation. This
        delegator preserves the ExecutionManager entrypoint used by
        ``execute()`` and the PP microbatch path.
        """
        return self.sampler_runtime.sample_tokens(num_tokens, padded_num_reqs, **kwargs)

    # Compile orchestration moved to CompileOrchestrator (phase 11.4). The
    # delegators below keep ExecutionManager's entrypoints stable: the runner
    # calls compile() with this exact signature, and the orchestrator routes
    # per-variant compiles back through these names so instance-level
    # overrides (tests) still intercept them.

    def _get_compile_orchestrator(self) -> CompileOrchestrator:
        """Return the compile orchestrator, creating it on first use.

        Lazy so ExecutionManager instances built without ``__init__`` (tests
        construct via ``__new__``) still route ``compile()`` correctly.
        """
        orchestrator = self.__dict__.get("_compile_orchestrator")
        if orchestrator is None:
            orchestrator = CompileOrchestrator(self)
            self._compile_orchestrator = orchestrator
        return orchestrator

    _get_feasible_compile_pairs = staticmethod(CompileOrchestrator._get_feasible_compile_pairs)

    def compile(
        self,
        num_tokens_paddings: list[int],
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
        num_reqs_paddings: list[int] | None = None,
        prune_infeasible_pairs: bool = True,
    ) -> None:
        """Delegate to :meth:`CompileOrchestrator.compile` (phase 11.4).

        See the orchestrator for the full bucket-grid planning documentation.
        The signature is kept identical because the runner calls
        ``executor_manager.compile(...)`` directly.
        """
        self._get_compile_orchestrator().compile(
            num_tokens_paddings=num_tokens_paddings,
            num_reqs_max_model_len=num_reqs_max_model_len,
            max_pages_per_req=max_pages_per_req,
            max_num_reqs=max_num_reqs,
            metadata=metadata,
            num_reqs_paddings=num_reqs_paddings,
            prune_infeasible_pairs=prune_infeasible_pairs,
        )

    def _require_precompiled_variants(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        sampler_padded_num_reqs: int,
        use_pipeline_runtime: bool = True,
    ) -> None:
        """Validate that ``compile()`` already prepared the exact runtime bucket."""
        missing: list[str] = []
        if not self.model.mesh.is_mpmd:
            if not self._model_executor.has_model_step(int(num_tokens), int(padded_num_reqs)):
                missing.append(f"SPMD model_step({int(num_tokens)}, {int(padded_num_reqs)})")
        else:
            has_fused = bool(
                self._should_use_fused_pipeline_model_step(
                    num_tokens=int(num_tokens),
                    padded_num_reqs=int(padded_num_reqs),
                )
                and self._model_executor.has_model_step(int(num_tokens), int(padded_num_reqs))
            )
            has_split = bool(
                self._has_backbone_variant(
                    int(num_tokens),
                    use_pipeline_runtime=bool(use_pipeline_runtime),
                )
                and self._model_executor.has_lm_head(int(padded_num_reqs))
            )
            if not (has_fused or has_split):
                missing.append(
                    "MPMD model bucket "
                    f"tokens={int(num_tokens)} reqs={int(padded_num_reqs)} "
                    f"use_pipeline_runtime={bool(use_pipeline_runtime)}"
                )

        sampler_key = self._sampler_executor.cache_key(padded_num_reqs=int(sampler_padded_num_reqs))
        if not self._sampler_executor.has(sampler_key):
            missing.append(f"sampler({int(sampler_padded_num_reqs)})")
        greedy_sampler_key = self._sampler_executor.cache_key(
            padded_num_reqs=int(sampler_padded_num_reqs),
            greedy=True,
        )
        if not self._sampler_executor.has(greedy_sampler_key):
            missing.append(f"greedy_sampler({int(sampler_padded_num_reqs)})")
        if missing:
            raise RuntimeError(
                "Missing precompiled eSurge bucket(s): "
                + ", ".join(missing)
                + ". Runtime compilation is disabled; include these buckets in compile()."
            )

    def _compile_model_step_variant(self, **kwargs) -> None:
        """Delegate to :meth:`CompileOrchestrator._compile_model_step_variant`."""
        self._get_compile_orchestrator()._compile_model_step_variant(**kwargs)

    def _compile_pipeline_model_step_variant(self, **kwargs) -> None:
        """Delegate to :meth:`CompileOrchestrator._compile_pipeline_model_step_variant`."""
        self._get_compile_orchestrator()._compile_pipeline_model_step_variant(**kwargs)

    def _compile_backbone_variant(self, **kwargs) -> None:
        """Delegate to :meth:`CompileOrchestrator._compile_backbone_variant`."""
        self._get_compile_orchestrator()._compile_backbone_variant(**kwargs)

    def _reset_kv_pages_after_compile_warmup(self) -> None:
        """Delegate to :meth:`CompileOrchestrator._reset_kv_pages_after_compile_warmup`."""
        self._get_compile_orchestrator()._reset_kv_pages_after_compile_warmup()

    def _compile_lm_head_variant(self, **kwargs) -> None:
        """Delegate to :meth:`CompileOrchestrator._compile_lm_head_variant`."""
        self._get_compile_orchestrator()._compile_lm_head_variant(**kwargs)

    def _compile_sampler_variant(self, **kwargs) -> None:
        """Delegate to :meth:`CompileOrchestrator._compile_sampler_variant`."""
        self._get_compile_orchestrator()._compile_sampler_variant(**kwargs)

    def get_compile_configurations(
        self,
        kv_pages: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
        rng_key: jax.random.PRNGKey,
        num_tokens: int,
        max_num_reqs: int,
        padded_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
    ):
        """Generate compilation arguments for step function.

        Creates dummy input structures with correct shapes, dtypes, and shardings
        for tracing the step function during AOT/JIT compilation. All arrays are
        device-resident with appropriate sharding annotations to prevent XLA from
        generating multiple compilation variants.

        Args:
            kv_pages: KV cache pages (used as-is in compilation args).
            rng_key: Random key for sampling (device-placed with empty sharding).
            num_tokens: Token count (unused, for API compatibility).
            num_reqs_max_model_len: Max requests at model length (unused).
            max_pages_per_req: Max pages per request (unused).
            max_num_reqs: Maximum concurrent requests for buffer sizing.
            padded_num_reqs: Target padded request count for this compilation variant.
            metadata: KV cache metadata for buffer initialization.

        Returns:
            List of compilation arguments: [graphdef, graphstate, graphother, inputs]
            where inputs is a StepFunctionInputs PyTree with dummy values.

        Note:
            Dummy values use simple patterns (ones, zeros) since compilation only
            traces shapes/dtypes. The returned structures must match runtime
            shardings exactly to avoid recompilation.
        """

        temp_buffer = SequenceBuffer(
            max_num_reqs=max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            vocab_size=self.model.config.get_text_config().vocab_size,
            page_sizes=[metadata.page_size],
            sharding=self._empty_sharding,
        )

        scheduled_full_cpu = numpy.zeros((max_num_reqs,), dtype=numpy.int32)
        active_mask_full_cpu = numpy.zeros((max_num_reqs,), dtype=bool)
        window_row_indices_cpu = numpy.arange(max_num_reqs, dtype=numpy.int32)
        recurrent_slot_indices_cpu = numpy.arange(max_num_reqs, dtype=numpy.int32)
        # Ensure the dummy schedule never exceeds the token bucket used for this
        # compilation variant (otherwise CPU batch-prep will correctly reject it).
        active_reqs = max(1, min(padded_num_reqs, max_num_reqs, num_tokens))
        scheduled_full_cpu[:active_reqs] = 1
        active_mask_full_cpu[:active_reqs] = True
        input_ids_buf = jax.device_put(jnp.zeros((self.max_num_tokens,), dtype=jnp.int32), self._empty_sharding)
        position_ids_buf = jax.device_put(jnp.zeros((self.max_num_tokens,), dtype=jnp.int32), self._empty_sharding)

        mrope_position_ids_cpu = None
        prefill_embeds_cpu = None
        prefill_embeds_mask_cpu = None
        visual_pos_masks_cpu = None
        deepstack_visual_embeds_cpu = None

        cfg = getattr(self.model, "config", None)
        task_type = getattr(self.model, "_task_type", None)
        is_vlm_model = task_type == "image-text-to-text" or (
            cfg is not None
            and (getattr(cfg, "image_token_id", None) is not None or getattr(cfg, "video_token_id", None) is not None)
            and callable(getattr(self.model, "get_image_features", None))
        )
        if int(getattr(self.metadata, "data_parallel_size", 1) or 1) > 1:
            is_vlm_model = False
        uses_mrope_model = is_vlm_model and model_uses_mrope(self.model)

        if is_vlm_model:
            hidden_size = int(getattr(self.model.config.get_text_config(), "hidden_size", 0) or 1)
            prefill_embeds_cpu = numpy.zeros((int(num_tokens), hidden_size), dtype=numpy.float16)
            prefill_embeds_mask_cpu = numpy.zeros((int(num_tokens),), dtype=bool)
            if uses_mrope_model:
                mrope_position_ids_cpu = numpy.zeros((3, int(num_tokens)), dtype=numpy.int32)
                deepstack_indexes = getattr(
                    getattr(self.model.config, "vision_config", None), "deepstack_visual_indexes", None
                )
                deepstack_layers = len(deepstack_indexes) if deepstack_indexes else 0
                if deepstack_layers:
                    visual_pos_masks_cpu = numpy.zeros((int(num_tokens),), dtype=bool)
                    deepstack_visual_embeds_cpu = [
                        numpy.zeros((int(num_tokens), hidden_size), dtype=numpy.float16) for _ in range(deepstack_layers)
                    ]

        page_table_cpu_dummy = temp_buffer.page_table[0].page_table_cpu

        (
            dummy_metadata,
            input_ids_buf,
            position_ids_buf,
            scheduled_full,
            active_mask_full,
        ) = self._batch_preparer.prepare_batch_metadata(
            num_tokens_static=num_tokens,
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            token_ids_cpu=temp_buffer.token_ids,  # NumPy arrays from SequenceBuffer
            num_computed_tokens_cpu=temp_buffer.num_computed_tokens,
            temperature_cpu=temp_buffer.temperature,
            top_p_cpu=temp_buffer.top_p,
            top_k_cpu=temp_buffer.top_k,
            min_p_cpu=temp_buffer.min_p,
            frequency_penalties_cpu=temp_buffer.frequency_penalties,
            presence_penalties_cpu=temp_buffer.presence_penalties,
            repetition_penalties_cpu=temp_buffer.repetition_penalties,
            page_table_cpu=page_table_cpu_dummy,
            padded_num_reqs_in=padded_num_reqs,
            window_row_indices_cpu=window_row_indices_cpu,
            recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
            mrope_position_ids_cpu=mrope_position_ids_cpu,
            prefill_embeds_cpu=prefill_embeds_cpu,
            prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
            visual_pos_masks_cpu=visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
        )

        inputs = StepFunctionInputs(
            kv_pages=kv_pages,
            scheduled_full=scheduled_full,
            req_num_tokens_full=jax.device_put(jnp.full((max_num_reqs,), 10, dtype=jnp.int32), self._empty_sharding),
            active_mask_full=active_mask_full,
            rng_key=jax.device_put(rng_key, self._empty_sharding),
            batch_metadata=dummy_metadata,
        )

        return [self.graphdef, self.graphstate, self.graphother, inputs]
