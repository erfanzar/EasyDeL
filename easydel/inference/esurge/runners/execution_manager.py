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

import hashlib
import os
import threading
import time
import typing
import typing as tp
from functools import partial

import jax
import numpy
import spectrax as spx
from eformer.loggings import ProgressLogger, get_logger
from eformer.pytree import key_path_to_str
from ejkernel.ops import forward_autotune_only  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.experimental import multihost_utils

from easydel.caching import (
    HybridCache,
    ParallelHybridCacheView,
    RaggedPagesCache,
    RaggedPagesCacheConfig,
    RecurrentCacheView,
    UnifiedAttentionCache,
    UnifiedAttentionCacheConfig,
)
from easydel.infra.sharding import replicate_on_array_mesh, replicated_named_sharding
from easydel.layers.quantization import TurboQuantConfig

from ..core.sampler import build_history_token_counts
from ..utils import model_uses_mrope
from .async_types import DeviceInputTokenHandoff
from .execution_types import BatchMetadata, ModelStepOutputs, StepFunctionInputs
from .executors import BatchMetadataPreparer, ModelStepExecutor, SamplerExecutor
from .pipeline_plan import PipelineInferencePlan, metadata_num_pages, set_metadata_num_pages
from .sequence_buffer import SequenceBuffer

DEBUG_MODE = False

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule
    from easydel.infra.etils import MpMdSchedulers

logger = get_logger("eSurge-ExecutionManager")

# Syncing inputs after host->device metadata transfer makes `prep_time` more accurate,
# but it adds a device round-trip that hurts throughput. Keep it opt-in.
SYNC_INPUTS_FOR_TIMING = bool(int(os.environ.get("EASURGE_SYNC_INPUTS_FOR_TIMING", "0")))


class _PipelineMicrobatchScratchSlot(tp.TypedDict):
    """Reusable host buffers for one PP decode microbatch.

    eSurge builds decode microbatches by slicing an active request window into
    several smaller logical batches. Allocating fresh NumPy arrays on every
    token would show up directly in the decode loop, so each slot owns one set
    of CPU buffers and is repopulated in-place for the next step.
    """

    scheduled: numpy.ndarray
    active: numpy.ndarray
    token_ids: numpy.ndarray
    num_computed: numpy.ndarray
    temperature: numpy.ndarray
    top_p: numpy.ndarray
    top_k: numpy.ndarray
    min_p: numpy.ndarray
    frequency: numpy.ndarray
    presence: numpy.ndarray
    repetition: numpy.ndarray
    page_table: numpy.ndarray
    prev_count: int


_PipelineExecuteResult = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, tp.Any],
]


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int, min_input_pad: int) -> int:
    """Calculate padded request count for compilation efficiency.

    Pads the number of requests to powers of 2 (up to min_input_pad) or the nearest
    power of 2 above min_input_pad. This reduces the number of unique compilations
    needed while maintaining good utilization.

    Args:
        x: Actual number of requests to pad.
        upper_limit: Maximum allowed requests, acts as a cap on the returned value.
        min_input_pad: Minimum padding value to use when x is small.

    Returns:
        Padded request count, capped at upper_limit.

    Examples:
        >>> _get_padded_num_reqs_with_upper_limit(3, 32, 8)   # Returns 8
        >>> _get_padded_num_reqs_with_upper_limit(10, 32, 8)  # Returns 16
        >>> _get_padded_num_reqs_with_upper_limit(20, 16, 8)  # Returns 16

    Note:
        This function helps reduce JAX compilation overhead by bucketing
        request counts into a smaller set of sizes.
    """
    res = min_input_pad if x <= min_input_pad else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def _device_put_tree_uniform(tree, sharding):  # pyright: ignore[reportUnusedFunction]
    """Place every array leaf of ``tree`` on devices with the same sharding."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, sharding) if hasattr(x, "dtype") else x,
        tree,
    )


@partial(jax.jit, static_argnames=("padded_num_reqs",))
def _combine_pipeline_hidden_parts(
    hidden_parts: tuple[jax.Array, ...],
    logits_index_parts: tuple[jax.Array, ...],
    position_parts: tuple[jax.Array, ...],
    *,
    padded_num_reqs: int,
) -> jax.Array:
    with jax.named_scope("easydel/esurge/pp_hidden_parts/combine"):
        hidden = hidden_parts[0]
        combined = jnp.zeros((int(padded_num_reqs), int(hidden.shape[-1])), dtype=hidden.dtype)
        for part_hidden, part_indices, part_positions in zip(
            hidden_parts,
            logits_index_parts,
            position_parts,
            strict=True,
        ):
            combined = combined.at[part_positions].set(part_hidden[part_indices])
        return combined


def _tree_hash(tree):
    """Compute a hash tree for debugging structure/shape/dtype changes.

    Creates a PyTree with the same structure where each leaf is replaced
    by a hash string encoding its type, shape, dtype, and sharding.

    Args:
        tree: PyTree to hash.

    Returns:
        PyTree with same structure where leaves are hash strings encoding
        their original type, shape, dtype, and sharding information.

    Note:
        This is used for debugging recompilation issues by comparing hash
        trees between compilation and execution to detect structural changes.
    """

    def _map(p, x):
        p = key_path_to_str(p)
        maybe_info = (
            f"-{type(x)}"
            + "-"
            + str(getattr(x, "shape", "None"))
            + "-"
            + str(getattr(x, "dtype", "None"))
            + "-"
            + str(getattr(x, "sharding", "None"))
        )
        if isinstance(x, int | float | bool):
            maybe_info = f"-{x}"
        return (
            hashlib.md5(
                str(
                    p
                    + str(type(x))
                    + str(getattr(x, "shape", "None"))
                    + str(getattr(x, "dtype", "None"))
                    + str(getattr(x, "sharding", "None"))
                ).encode()
            ).hexdigest()
            + maybe_info
        )

    return jax.tree_util.tree_map_with_path(
        _map,
        tree,
        is_leaf=lambda x: isinstance(
            x,
            jax.Array | numpy.ndarray | int | float | bool | None,
        ),
    )


def _tree_hash_diff(orgin, new):
    """Compare two hash trees and print differences.

    Compares hash trees created by _tree_hash() and prints any paths
    where the hashes differ, helping debug unexpected recompilations.

    Args:
        orgin: Original hash tree (typically from compilation).
        new: New hash tree (typically from execution).

    Returns:
        PyTree of booleans indicating whether each leaf matches.
    """

    def _map(p, t1, t2):
        p = key_path_to_str(p)
        oo = t1 == t2
        if not oo:
            print(f"path: {p} out: {oo} orgin: {t1} new: {t2}")
        return oo

    return jax.tree_util.tree_map_with_path(_map, orgin, new, is_leaf=lambda x: isinstance(x, str))


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
        self.mpmd_scheduler = mpmd_scheduler
        self.pipeline_plan = pipeline_plan
        self.pp_microbatch_count = pp_microbatch_count
        self.pp_microbatch_size = pp_microbatch_size
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

        # Prefer HybridCache (per-operation cache views) as the universal container.
        # Keep paged-cache parameters consistent with the scheduler config.
        self.kv_pages = self._init_operations_cache_with_retry(
            quantizer=quantizer,
            masking_details=getattr(text_config, "get_mask_details", lambda: None)(),
        )

        self.graphdef, self.graphstate, self.graphother = model.split_module()

        self.log_it = logger.info if verbose else logger.debug
        self._verbose = verbose

        self._empty_sharding = replicated_named_sharding(model.mesh)

        self._sampler_sharding = self._empty_sharding
        self.rng_key = jax.device_put(jax.random.PRNGKey(0), self._sampler_sharding)

        self._debug_baselines = {}
        self._runtime_compile_lock = threading.RLock()
        self._runtime_compile_metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig | None = None
        self._runtime_compile_max_num_reqs: int | None = None
        self._runtime_lazy_compile = True

        self._batch_preparer = BatchMetadataPreparer(
            metadata=self.metadata,
            empty_sharding=self._empty_sharding,
            max_num_tokens=self.max_num_tokens,
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            min_input_pad=self.min_input_pad,
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
        )
        self._sampler_vocab_size = int(self.model.config.get_text_config().vocab_size)
        self._sampler_executor = SamplerExecutor(
            model=self.model,
            max_model_len=self.max_model_len,
            empty_sharding=self._sampler_sharding,
            use_aot_forward=self.use_aot_forward,
        )
        self._sampler_min_input_pad = 1
        self._sampler_zero_token_counts = jnp.zeros(
            (self.max_num_reqs, self._sampler_vocab_size),
            dtype=jnp.uint32,
            out_sharding=self._sampler_sharding,
        )
        self._req_num_tokens_placeholder = jnp.zeros(
            (self.max_num_reqs,),
            dtype=jnp.int32,
            out_sharding=self._empty_sharding,
        )
        self._sampler_zero_window_row_indices = jnp.zeros(
            (self.max_num_reqs,),
            dtype=jnp.int32,
            out_sharding=self._sampler_sharding,
        )
        self._model_num_tokens_paddings: list[int] = []
        self._sampler_token_counts = self._sampler_zero_token_counts
        self._sampler_penalty_state_dirty = True
        self._sampler_penalty_state_ready = False
        self._sampler_penalty_rebuild_token_ids_cpu: numpy.ndarray | None = None
        self._sampler_penalty_rebuild_seq_lens_cpu: numpy.ndarray | None = None
        self._sampler_gather_positions_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_sampling_seeds_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_scatter_positions_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_window_row_indices_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_scheduled_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_seq_lens_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_active_mask_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.bool_)
        self._sampler_temperature_cpu = numpy.ones((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_top_p_cpu = numpy.ones((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_top_k_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_min_p_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_frequency_penalties_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_presence_penalties_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_repetition_penalties_cpu = numpy.ones((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_packed_i32_cpu_by_reqs: dict[int, numpy.ndarray] = {}
        self._sampler_packed_f32_cpu_by_reqs: dict[int, numpy.ndarray] = {}
        self._sampler_packed_misc_i32_cpu = numpy.zeros((2,), dtype=numpy.int32)
        self._sampler_prefix_cpu_by_reqs: dict[int, numpy.ndarray] = {}
        self._pipeline_microbatch_scratch: list[_PipelineMicrobatchScratchSlot] = []
        self._pipeline_microbatch_scratch_signature: tuple[tuple[tuple[int, ...], str], ...] | None = None
        self._pipeline_logits_index_cache: dict[tuple[int, str], jax.Array] = {}
        self._pipeline_handoff_scalar_cache: dict[tuple[int, int], tuple[jax.Array, jax.Array]] = {}

        @jax.jit
        def _rebuild_penalty_counts(token_history: jax.Array, seq_lens: jax.Array) -> jax.Array:
            with jax.named_scope("easydel/esurge/sampler/rebuild_penalty_counts"):
                return build_history_token_counts(
                    token_history=token_history,
                    seq_lens=seq_lens.astype(jnp.int32),
                    active_mask=seq_lens > 0,
                    vocab_size=self._sampler_vocab_size,
                )

        self._rebuild_penalty_counts = _rebuild_penalty_counts

        @partial(jax.jit, static_argnames=("padded_num_reqs",))
        def _scatter_sampler_outputs(
            sampled_tokens: jax.Array,
            valid_mask: jax.Array,
            scatter_positions: jax.Array,
            padded_num_reqs: int,
        ) -> tuple[jax.Array, jax.Array]:
            with jax.named_scope("easydel/esurge/sampler/scatter_outputs"):
                spill = int(scatter_positions.shape[0])
                full_tokens = jnp.full((int(padded_num_reqs) + spill,), -1, dtype=sampled_tokens.dtype)
                full_valid = jnp.zeros((int(padded_num_reqs) + spill,), dtype=jnp.bool_)
                full_tokens = full_tokens.at[scatter_positions].set(jnp.where(valid_mask, sampled_tokens, -1))
                full_valid = full_valid.at[scatter_positions].set(valid_mask)
                return full_tokens[:padded_num_reqs], full_valid[:padded_num_reqs]

        self._scatter_sampler_outputs = _scatter_sampler_outputs

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
            return self.model.init_operations_cache(
                batch_size=int(self.max_num_reqs),
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
        self._pipeline_microbatch_scratch.clear()
        self._pipeline_microbatch_scratch_signature = None
        self._pipeline_logits_index_cache.clear()
        self._pipeline_handoff_scalar_cache.clear()

    def invalidate_sampler_penalty_state(
        self,
        token_ids_cpu: numpy.ndarray | None = None,
        seq_lens_cpu: numpy.ndarray | None = None,
    ) -> None:
        """Mark sampler penalty counters dirty after host-side row reorderings.

        The frequency / presence / repetition penalty kernels keep an
        incremental device-side count of how often each token has appeared
        per request. That counter must be rebuilt whenever the runner
        permutes rows in :class:`SequenceBuffer` (e.g. swap_rows /
        condense). This method records the new ground-truth host views
        and flips the dirty flag so the next sampler call goes through
        :meth:`_ensure_sampler_penalty_state`.

        Args:
            token_ids_cpu: ``(max_num_reqs, max_model_len)`` host array of
                token ids per slot — the source of truth used during the
                next rebuild. ``None`` keeps the previously-recorded view.
            seq_lens_cpu: ``(max_num_reqs,)`` host array of effective
                sequence lengths. ``None`` keeps the previous view.
        """
        if token_ids_cpu is not None:
            self._sampler_penalty_rebuild_token_ids_cpu = token_ids_cpu
        if seq_lens_cpu is not None:
            self._sampler_penalty_rebuild_seq_lens_cpu = seq_lens_cpu
        self._sampler_penalty_state_dirty = True
        self._sampler_penalty_state_ready = False

    def _ensure_sampler_penalty_state(self) -> None:
        """Recompute device-side per-token frequency counts when dirty.

        No-op when the cached counters are clean. On dirty, takes the
        host views captured by :meth:`invalidate_sampler_penalty_state`,
        broadcasts them across ranks under multi-host JAX, transfers
        them onto the sampler sharding, and runs ``_rebuild_penalty_counts``
        to rebuild the exact device-side per-token counts. Toggles the
        dirty / ready flags on success.

        Raises:
            RuntimeError: If a rebuild is requested without a previously
                recorded host source — indicates a missing
                :meth:`invalidate_sampler_penalty_state` call earlier in the
                step.
        """
        if self._sampler_penalty_state_ready and not self._sampler_penalty_state_dirty:
            return
        if self._sampler_penalty_rebuild_token_ids_cpu is None or self._sampler_penalty_rebuild_seq_lens_cpu is None:
            raise RuntimeError("Sampler penalty state rebuild requested without a full-sequence source.")

        _token_ids_cpu = self._sampler_penalty_rebuild_token_ids_cpu
        _seq_lens_cpu = self._sampler_penalty_rebuild_seq_lens_cpu
        if jax.process_count() > 1:
            _token_ids_cpu, _seq_lens_cpu = multihost_utils.broadcast_one_to_all((_token_ids_cpu, _seq_lens_cpu))
        token_history = jax.device_put(_token_ids_cpu, self._sampler_sharding)
        seq_lens = jax.device_put(_seq_lens_cpu, self._sampler_sharding)
        self._sampler_token_counts = self._rebuild_penalty_counts(token_history, seq_lens)
        self._sampler_penalty_state_dirty = False
        self._sampler_penalty_state_ready = True

    def _prepare_compact_sampler_window(
        self,
        *,
        padded_num_reqs: int,
        scheduled_full_cpu: numpy.ndarray,
        active_mask_full_cpu: numpy.ndarray,
        window_row_indices_cpu: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        temperature_cpu: numpy.ndarray,
        top_p_cpu: numpy.ndarray,
        top_k_cpu: numpy.ndarray,
        min_p_cpu: numpy.ndarray,
        frequency_penalties_cpu: numpy.ndarray,
        presence_penalties_cpu: numpy.ndarray,
        repetition_penalties_cpu: numpy.ndarray,
    ) -> tuple[int, int, int]:
        """Compact the sampler workload to rows that can actually emit tokens.

        The model forward may still need a wider request window to preserve
        sparse row layout, especially for async scheduling. The sampler only
        needs rows that are both active and scheduled, so compacting here
        avoids burning top-k/top-p work on zero-token rows.
        """
        padded_num_reqs = int(padded_num_reqs)
        scheduled_window = numpy.asarray(scheduled_full_cpu[:padded_num_reqs], dtype=numpy.int32)
        active_window = numpy.asarray(active_mask_full_cpu[:padded_num_reqs], dtype=numpy.bool_)
        sample_positions = numpy.flatnonzero(active_window & (scheduled_window > 0))
        sample_count = int(sample_positions.size)
        if sample_count <= 0:
            raise RuntimeError("Sampler compaction found no scheduled active rows for a non-empty execution window.")

        sampler_padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            sample_count,
            upper_limit=padded_num_reqs,
            min_input_pad=int(self._sampler_min_input_pad),
        )

        gather_positions = self._sampler_gather_positions_cpu
        sampling_seeds = self._sampler_sampling_seeds_cpu
        scatter_positions = self._sampler_scatter_positions_cpu
        window_rows = self._sampler_window_row_indices_cpu
        scheduled_out = self._sampler_scheduled_cpu
        seq_lens_out = self._sampler_seq_lens_cpu
        active_out = self._sampler_active_mask_cpu
        temperature_out = self._sampler_temperature_cpu
        top_p_out = self._sampler_top_p_cpu
        top_k_out = self._sampler_top_k_cpu
        min_p_out = self._sampler_min_p_cpu
        frequency_out = self._sampler_frequency_penalties_cpu
        presence_out = self._sampler_presence_penalties_cpu
        repetition_out = self._sampler_repetition_penalties_cpu

        gather_positions[:sampler_padded_num_reqs] = 0
        window_rows[:sampler_padded_num_reqs] = 0
        scheduled_out[:sampler_padded_num_reqs] = 0
        seq_lens_out[:sampler_padded_num_reqs] = 0
        active_out[:sampler_padded_num_reqs] = False
        temperature_out[:sampler_padded_num_reqs] = 1.0
        top_p_out[:sampler_padded_num_reqs] = 1.0
        top_k_out[:sampler_padded_num_reqs] = 0
        min_p_out[:sampler_padded_num_reqs] = 0.0
        frequency_out[:sampler_padded_num_reqs] = 0.0
        presence_out[:sampler_padded_num_reqs] = 0.0
        repetition_out[:sampler_padded_num_reqs] = 1.0

        padding_range = numpy.arange(sampler_padded_num_reqs, dtype=numpy.int32)
        sampling_seeds[:sampler_padded_num_reqs] = padded_num_reqs + padding_range
        scatter_positions[:sampler_padded_num_reqs] = padded_num_reqs + padding_range

        gather_positions[:sample_count] = sample_positions
        sampling_seeds[:sample_count] = sample_positions
        scatter_positions[:sample_count] = sample_positions
        window_rows[:sample_count] = window_row_indices_cpu[sample_positions]
        scheduled_out[:sample_count] = scheduled_window[sample_positions]
        seq_lens_out[:sample_count] = (
            numpy.asarray(num_computed_tokens_cpu[:padded_num_reqs], dtype=numpy.int32)[sample_positions]
            + scheduled_window[sample_positions]
        )
        active_out[:sample_count] = True
        temperature_out[:sample_count] = numpy.asarray(temperature_cpu[:padded_num_reqs], dtype=numpy.float32)[
            sample_positions
        ]
        top_p_out[:sample_count] = numpy.asarray(top_p_cpu[:padded_num_reqs], dtype=numpy.float32)[sample_positions]
        top_k_out[:sample_count] = numpy.asarray(top_k_cpu[:padded_num_reqs], dtype=numpy.int32)[sample_positions]
        min_p_out[:sample_count] = numpy.asarray(min_p_cpu[:padded_num_reqs], dtype=numpy.float32)[sample_positions]
        frequency_out[:sample_count] = numpy.asarray(frequency_penalties_cpu[:padded_num_reqs], dtype=numpy.float32)[
            sample_positions
        ]
        presence_out[:sample_count] = numpy.asarray(presence_penalties_cpu[:padded_num_reqs], dtype=numpy.float32)[
            sample_positions
        ]
        repetition_out[:sample_count] = numpy.asarray(
            repetition_penalties_cpu[:padded_num_reqs],
            dtype=numpy.float32,
        )[sample_positions]
        total_tokens = int(scheduled_out[:sample_count].sum())
        return sample_count, sampler_padded_num_reqs, total_tokens

    def has_compiled_variants(self) -> bool:
        """Check whether both model and sampler executors have compiled variants.

        Returns:
            True if both the model executor and the sampler executor have at
            least one cached compiled function, False otherwise.
        """
        return bool(self._model_executor.cache_keys()) and bool(self._sampler_executor.cache_keys())

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
            # Keep sub-executors in sync with the active model reference.
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

        # Clear cached baselines so future diagnostics re-hash with new weights.
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
        page_table_version: int | None = None,
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
        jax.Array,
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
            input_ids_buf: Contiguous token ID buffer [max_num_tokens]. Flattened
                across requests for efficient batch processing.
            position_ids_buf: Contiguous position ID buffer [max_num_tokens].
                Parallel to input_ids_buf with position indices.
            padded_num_reqs: Bucketed request count for compilation lookup. Must
                be a power of 2 (or min_input_pad) matching a compiled variant.

        Returns:
            Tuple of 7 elements:
                - out_tokens_full: Generated tokens [max_num_reqs], -1 for invalid.
                - valid_mask_full: Boolean mask for valid generations [max_num_reqs].
                - input_ids_buf: Updated input buffer (may contain new tokens).
                - position_ids_buf: Updated position buffer.
                - hidden_states: Last layer hidden states [num_tokens, hidden_dim].
                - logits: Output logits [padded_num_reqs, vocab_size].
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
        microbatch_result = self._try_execute_pipeline_microbatches(
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
            return microbatch_result

        start_prep = time.time()
        prep_phase_start = start_prep
        (
            batch_metadata,
            input_ids_buf,
            position_ids_buf,
            scheduled_full,
            active_mask_full,
        ) = self.prepare_batch_metadata(
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
        prep_handoff_took = 0.0
        if device_token_handoff is not None:
            prep_phase_start = time.time()
            batch_metadata = self._apply_device_token_handoff(batch_metadata, device_token_handoff)
            input_ids_buf = batch_metadata.input_ids_buf
            prep_handoff_took = time.time() - prep_phase_start
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
        if self.model.mesh.is_mpmd and self._model_executor.supports_pipeline_model_step:
            sampler_prefix = self._sampler_prefix_cpu_by_reqs.get(int(sampler_padded_num_reqs))
            if sampler_prefix is None:
                sampler_prefix = numpy.arange(int(sampler_padded_num_reqs), dtype=numpy.int32)
                self._sampler_prefix_cpu_by_reqs[int(sampler_padded_num_reqs)] = sampler_prefix
            if numpy.array_equal(
                self._sampler_gather_positions_cpu[: int(sampler_padded_num_reqs)],
                sampler_prefix,
            ):
                model_logits_padded_num_reqs = int(sampler_padded_num_reqs)
        prep_phase_start = time.time()
        self._ensure_runtime_variants(
            num_tokens=num_tokens,
            padded_num_reqs=model_logits_padded_num_reqs,
            sampler_padded_num_reqs=sampler_padded_num_reqs,
        )
        prep_ensure_variants_took = time.time() - prep_phase_start

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

        start_exec = time.time()
        model_outputs = self.execute_model(
            num_tokens=num_tokens,
            padded_num_reqs=model_logits_padded_num_reqs,
            inputs=inputs,
        )

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
            need_penalties=bool(
                numpy.any(frequency_penalties_cpu != 0.0)
                or numpy.any(presence_penalties_cpu != 0.0)
                or numpy.any(repetition_penalties_cpu != 1.0)
            ),
        )
        rng_key_out, out_tokens_full, valid_mask_full, token_counts_out = sampler_out
        self.rng_key = rng_key_out
        self._sampler_token_counts = token_counts_out
        if wait_for_outputs:
            jax.block_until_ready(model_outputs.logits)
            exec_took = time.time() - start_exec

            start_sample = time.time()
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
            "sample_time": sample_took,
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

    @staticmethod
    def _apply_device_token_handoff(
        batch_metadata: BatchMetadata,
        handoff: DeviceInputTokenHandoff,
    ) -> BatchMetadata:
        """Attach a compiled-step token handoff to batch metadata.

        PP async decode has a one-token loop-carried dependency: token ``N``
        sampled by the final stage is token ``N+1``'s input on stage 0. The old
        path resolved that dependency by materializing token ``N`` on the host
        and repairing the CPU sequence buffer before the next launch. This
        helper keeps the dependency on device by carrying the sampled-token
        arrays into :class:`BatchMetadata`; the compiled model step resolves
        placeholders via ``BatchMetadata.model_input_ids`` inside the JIT region.

        The returned :class:`BatchMetadata` is a fresh PyTree with all metadata
        leaves preserved except the handoff buffers. CPU request state is
        repaired separately after dispatch, which keeps scheduler/accounting
        correctness while removing host token materialization from the
        stage-launch critical path.
        """
        return BatchMetadata(
            packed_qsl_seqlens=batch_metadata.packed_qsl_seqlens,
            packed_i32_padded=batch_metadata.packed_i32_padded,
            packed_f32_padded=batch_metadata.packed_f32_padded,
            packed_misc_i32=batch_metadata.packed_misc_i32,
            pages_tables=batch_metadata.pages_tables,
            input_ids_buf=batch_metadata.input_ids_buf,
            position_ids_buf=batch_metadata.position_ids_buf,
            input_token_handoff_positions=handoff.input_positions,
            input_token_handoff_ids=handoff.token_ids,
            input_token_handoff_count=handoff.count,
            input_token_handoff_offset=(
                handoff.offset if handoff.offset is not None else jnp.zeros((), dtype=jnp.int32)
            ),
            num_tokens=batch_metadata.num_tokens,
            slot_mapping=batch_metadata.slot_mapping,
            num_kv_update_slices=batch_metadata.num_kv_update_slices,
            pixel_values=batch_metadata.pixel_values,
            image_grid_thw=batch_metadata.image_grid_thw,
            pixel_values_videos=batch_metadata.pixel_values_videos,
            video_grid_thw=batch_metadata.video_grid_thw,
            mrope_position_ids=batch_metadata.mrope_position_ids,
            prefill_embeds=batch_metadata.prefill_embeds,
            prefill_embeds_mask=batch_metadata.prefill_embeds_mask,
            visual_pos_masks=batch_metadata.visual_pos_masks,
            deepstack_visual_embeds=batch_metadata.deepstack_visual_embeds,
        )

    def _get_pipeline_handoff_scalars(self, *, offset: int, count: int) -> tuple[jax.Array, jax.Array]:
        """Return cached device scalars for a PP microbatch handoff span.

        The exact decode microbatch path uses the same full-window sampled-token
        handoff for every microbatch. Only the live slice changes. Creating
        those scalar leaves once avoids putting tiny host arrays on every token,
        while also avoiding the much more expensive device scatter that used to
        build a fresh handoff vector per microbatch.
        """
        key = (int(offset), int(count))
        cached = self._pipeline_handoff_scalar_cache.get(key)
        if cached is not None:
            return cached
        offset_dev = jax.device_put(numpy.asarray(int(offset), dtype=numpy.int32), self._empty_sharding)
        count_dev = jax.device_put(numpy.asarray(int(count), dtype=numpy.int32), self._empty_sharding)
        cached = (offset_dev, count_dev)
        self._pipeline_handoff_scalar_cache[key] = cached
        return cached

    def _ensure_pipeline_microbatch_scratch(
        self,
        count: int,
        *,
        scheduled_full_cpu: numpy.ndarray,
        active_mask_full_cpu: numpy.ndarray,
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
    ) -> list[_PipelineMicrobatchScratchSlot]:
        """Return reusable CPU scratch slots matching the current window shapes.

        The signature is the shape/dtype contract for all host-side arrays used
        to build a decode microbatch. Any shape change invalidates the scratch
        pool; otherwise slots are reused in-place across decode iterations.
        """
        signature = tuple(
            (tuple(arr.shape), str(arr.dtype))
            for arr in (
                scheduled_full_cpu,
                active_mask_full_cpu,
                token_ids_cpu,
                num_computed_tokens_cpu,
                temperature_cpu,
                top_p_cpu,
                top_k_cpu,
                min_p_cpu,
                frequency_penalties_cpu,
                presence_penalties_cpu,
                repetition_penalties_cpu,
                page_table_cpu,
            )
        )
        if self._pipeline_microbatch_scratch_signature != signature:
            self._pipeline_microbatch_scratch = []
            self._pipeline_microbatch_scratch_signature = signature

        def _new_slot() -> _PipelineMicrobatchScratchSlot:
            return {
                "scheduled": numpy.zeros_like(scheduled_full_cpu),
                "active": numpy.zeros_like(active_mask_full_cpu),
                "token_ids": numpy.zeros_like(token_ids_cpu),
                "num_computed": numpy.zeros_like(num_computed_tokens_cpu),
                "temperature": numpy.ones_like(temperature_cpu),
                "top_p": numpy.ones_like(top_p_cpu),
                "top_k": numpy.zeros_like(top_k_cpu),
                "min_p": numpy.zeros_like(min_p_cpu),
                "frequency": numpy.zeros_like(frequency_penalties_cpu),
                "presence": numpy.zeros_like(presence_penalties_cpu),
                "repetition": numpy.ones_like(repetition_penalties_cpu),
                "page_table": numpy.zeros_like(page_table_cpu),
                "prev_count": 0,
            }

        while len(self._pipeline_microbatch_scratch) < int(count):
            self._pipeline_microbatch_scratch.append(_new_slot())
        return self._pipeline_microbatch_scratch[: int(count)]

    @staticmethod
    def _populate_pipeline_microbatch_scratch(
        slot: _PipelineMicrobatchScratchSlot,
        *,
        chunk: numpy.ndarray,
        scheduled_full_cpu: numpy.ndarray,
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
    ) -> tuple[
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
    ]:
        """Fill one scratch slot with a compact request chunk.

        ``chunk`` contains row indices from the full active window. The first
        ``len(chunk)`` rows of each scratch array become a dense microbatch;
        stale rows from the previous use of the slot are cleared so JAX sees a
        stable padded shape with no leftover request metadata.
        """
        real_count = len(chunk)
        prev_count = int(slot["prev_count"])
        clear_count = max(prev_count, real_count)

        mb_scheduled = slot["scheduled"]
        mb_active = slot["active"]
        mb_token_ids = slot["token_ids"]
        mb_num_computed = slot["num_computed"]
        mb_temperature = slot["temperature"]
        mb_top_p = slot["top_p"]
        mb_top_k = slot["top_k"]
        mb_min_p = slot["min_p"]
        mb_frequency = slot["frequency"]
        mb_presence = slot["presence"]
        mb_repetition = slot["repetition"]
        mb_page_table = slot["page_table"]

        if clear_count:
            mb_scheduled[:clear_count] = 0
            mb_active[:clear_count] = False
            mb_num_computed[:clear_count] = 0
            mb_temperature[:clear_count] = 1.0
            mb_top_p[:clear_count] = 1.0
            mb_top_k[:clear_count] = 0
            mb_min_p[:clear_count] = 0.0
            mb_frequency[:clear_count] = 0.0
            mb_presence[:clear_count] = 0.0
            mb_repetition[:clear_count] = 1.0
        if prev_count > real_count:
            mb_token_ids[real_count:prev_count] = 0
            mb_page_table[real_count:prev_count] = 0

        if real_count:
            mb_scheduled[:real_count] = scheduled_full_cpu[chunk]
            mb_active[:real_count] = True
            mb_token_ids[:real_count] = token_ids_cpu[chunk]
            mb_num_computed[:real_count] = num_computed_tokens_cpu[chunk]
            mb_temperature[:real_count] = temperature_cpu[chunk]
            mb_top_p[:real_count] = top_p_cpu[chunk]
            mb_top_k[:real_count] = top_k_cpu[chunk]
            mb_min_p[:real_count] = min_p_cpu[chunk]
            mb_frequency[:real_count] = frequency_penalties_cpu[chunk]
            mb_presence[:real_count] = presence_penalties_cpu[chunk]
            mb_repetition[:real_count] = repetition_penalties_cpu[chunk]
            mb_page_table[:real_count] = page_table_cpu[chunk]

        slot["prev_count"] = real_count
        return (
            mb_scheduled,
            mb_active,
            mb_token_ids,
            mb_num_computed,
            mb_temperature,
            mb_top_p,
            mb_top_k,
            mb_min_p,
            mb_frequency,
            mb_presence,
            mb_repetition,
            mb_page_table,
        )

    def _get_pipeline_logits_indices(self, count: int, like: jax.Array) -> jax.Array:
        """Return cached decode logits indices placed like a PP hidden-state part."""
        count = int(count)
        sharding_key = repr(like.sharding)
        key = (count, sharding_key)
        cached = self._pipeline_logits_index_cache.get(key)
        if cached is not None:
            return cached
        indices = jnp.arange(count, dtype=jnp.int32)
        indices = replicate_on_array_mesh(indices, like)
        self._pipeline_logits_index_cache[key] = indices
        return indices

    @staticmethod
    def _use_pipeline_microbatching(
        *,
        active_count: int,
        num_stages: int,
        microbatch_count: int,
        microbatch_req_count: int,
        min_token_bucket: int,
        has_compiled_handoff: bool,
    ) -> bool:
        """Choose PP wavefront microbatching when decode rows can fill stages.

        Decode microbatching overlaps pipeline stages by splitting active
        requests into a wavefront. One active request cannot fill a PP pipeline,
        but ``num_stages`` active decode requests can: while stage 3 finishes
        microbatch 0, stage 2 can process microbatch 1, and so on.

        Keep the fused full-window path for underfilled batches because it uses
        fewer launches and is better latency-wise. Without compiled token
        handoff, also require each microbatch to carry enough real decode rows
        to fill the smallest token bucket; otherwise the loop-carried sampled
        token dependency comes back as a large repair/synchronization before
        the next launch. With compiled handoff, a one-token microbatch is a
        legitimate PP wavefront unit because the previous sampled token is
        consumed inside the compiled model step.
        """
        if int(active_count) < int(num_stages):
            return False
        if int(microbatch_count) < int(num_stages):
            return False
        if bool(has_compiled_handoff):
            return int(microbatch_req_count) >= 1
        return int(microbatch_req_count) >= max(1, int(min_token_bucket))

    @staticmethod
    def _resolve_pipeline_microbatch_shape(
        *,
        active_count: int,
        num_stages: int,
        pp_microbatch_count: int | str | None,
        pp_microbatch_size: int | str | None,
    ) -> tuple[int, int] | None:
        """Resolve the PP wavefront shape for the current active decode rows.

        Returns ``(microbatch_count, rows_per_microbatch)``. ``None`` means the
        user disabled PP microbatch wavefronting for this runner.
        """
        def _policy_value(value: int | str | None) -> int | str | None:
            if value is None:
                return None
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered == "auto":
                    return "auto"
                if lowered in {"none", "off", "disable", "disabled"}:
                    return None
                value = int(lowered)
            value = int(value)
            return None if value == 0 else value

        active_count = int(active_count)
        num_stages = int(num_stages)
        pp_microbatch_count = _policy_value(pp_microbatch_count)
        pp_microbatch_size = _policy_value(pp_microbatch_size)
        if active_count <= 0 or num_stages <= 0:
            return None
        if pp_microbatch_count is None or pp_microbatch_size is None:
            return None

        count_is_auto = pp_microbatch_count == "auto"
        size_is_auto = pp_microbatch_size == "auto"
        if not count_is_auto and not size_is_auto:
            raise ValueError("Only one of pp_microbatch_count or pp_microbatch_size may be a positive integer.")

        if not size_is_auto:
            rows_per_microbatch = max(1, int(pp_microbatch_size))
            microbatch_count = (active_count + rows_per_microbatch - 1) // rows_per_microbatch
        elif not count_is_auto:
            microbatch_count = max(1, min(int(pp_microbatch_count), active_count))
            rows_per_microbatch = (active_count + microbatch_count - 1) // microbatch_count
        else:
            microbatch_count = min(num_stages, active_count)
            rows_per_microbatch = (active_count + microbatch_count - 1) // microbatch_count

        return int(microbatch_count), int(rows_per_microbatch)

    def _try_execute_pipeline_microbatches(
        self,
        *,
        num_tokens: int,
        scheduled_full_cpu: numpy.ndarray,
        active_mask_full_cpu: numpy.ndarray,
        window_row_indices_cpu: numpy.ndarray,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        padded_num_reqs: int,
        req_num_tokens_full_cpu: numpy.ndarray,
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
        mrope_position_ids_cpu: numpy.ndarray | None,
        prefill_embeds_cpu: numpy.ndarray | None,
        prefill_embeds_mask_cpu: numpy.ndarray | None,
        visual_pos_masks_cpu: numpy.ndarray | None,
        deepstack_visual_embeds_cpu: list[numpy.ndarray] | None,
        pixel_values: numpy.ndarray | None,
        image_grid_thw: numpy.ndarray | None,
        pixel_values_videos: numpy.ndarray | None,
        video_grid_thw: numpy.ndarray | None,
        device_token_handoff: DeviceInputTokenHandoff | None,
        wait_for_outputs: bool,
    ) -> _PipelineExecuteResult | None:
        """Try the SpectraX PP wavefront path for decode-only microbatches.

        Architecture:
            * The normal full-window PP path launches each physical stage once.
              That is best for small active windows because it minimizes launch
              count.
            * This path is selected only for larger decode windows. It splits
              active rows into same-shaped microbatches, asks SpectraX to launch
              them as a host wavefront, and carries each stage's local KV-cache
              leaves from microbatch N to N+1.
            * EasyDeL then gathers the sampled hidden states, runs the LM head,
              and scatters sampler output back to the original request rows.

        Returns:
            The same tuple shape as :meth:`execute` when the wavefront path is
            applicable; ``None`` when the caller should use the fused
            full-window path instead.
        """
        pipeline_plan = self._model_executor.pipeline_plan
        pipeline_runtime = self._model_executor.pipeline_runtime
        if pipeline_plan is None or not pipeline_plan.is_enabled or pipeline_runtime is None:
            return None
        if any(
            x is not None
            for x in (
                mrope_position_ids_cpu,
                prefill_embeds_cpu,
                prefill_embeds_mask_cpu,
                visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu,
                pixel_values,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
            )
        ):
            return None

        active_window = numpy.asarray(active_mask_full_cpu[:padded_num_reqs], dtype=numpy.bool_)
        scheduled_window = numpy.asarray(scheduled_full_cpu[:padded_num_reqs], dtype=numpy.int32)
        active_positions = numpy.flatnonzero(active_window & (scheduled_window > 0))
        active_count = int(active_positions.size)
        if active_count < 2:
            return None
        if not numpy.all(scheduled_window[active_positions] == 1):
            return None

        num_stages = len(pipeline_plan.stage_meshes)
        if num_stages < 2:
            return None
        microbatch_shape = self._resolve_pipeline_microbatch_shape(
            active_count=active_count,
            num_stages=num_stages,
            pp_microbatch_count=self.pp_microbatch_count,
            pp_microbatch_size=self.pp_microbatch_size,
        )
        if microbatch_shape is None:
            return None
        microbatch_count, microbatch_req_count = microbatch_shape
        if not self._use_pipeline_microbatching(
            active_count=active_count,
            num_stages=num_stages,
            microbatch_count=microbatch_count,
            microbatch_req_count=microbatch_req_count,
            min_token_bucket=int(self.min_input_pad),
            has_compiled_handoff=device_token_handoff is not None,
        ):
            return None
        microbatch_padded_reqs = _get_padded_num_reqs_with_upper_limit(
            microbatch_req_count,
            upper_limit=int(padded_num_reqs),
            min_input_pad=int(self.min_input_pad),
        )
        token_paddings = self._model_num_tokens_paddings
        if token_paddings:
            candidates = [int(x) for x in token_paddings if int(x) >= int(microbatch_req_count)]
            if not candidates:
                return None
            microbatch_num_tokens = min(candidates)
        else:
            microbatch_num_tokens = max(int(num_tokens), int(microbatch_req_count), int(self.min_input_pad))
        if not self._model_executor.has((microbatch_num_tokens, microbatch_padded_reqs, "model_step", "mpmd")):
            if self._runtime_lazy_compile:
                self._ensure_runtime_variants(
                    num_tokens=microbatch_num_tokens,
                    padded_num_reqs=padded_num_reqs,
                    sampler_padded_num_reqs=padded_num_reqs,
                    compile_split_pp_path=True,
                )
            if not self._model_executor.has_backbone(microbatch_num_tokens):
                return None
        if not self._model_executor.has_lm_head(padded_num_reqs):
            return None

        start_prep = time.time()
        chunks = [
            active_positions[i : i + microbatch_req_count]
            for i in range(0, active_count, microbatch_req_count)
        ]
        if len(chunks) < 2:
            return None
        if device_token_handoff is not None and int(device_token_handoff.token_ids.shape[0]) < active_count:
            return None

        input_batches: list[tuple[tp.Any, tp.Any, tp.Any, BatchMetadata]] = []
        metadata_batches: list[BatchMetadata] = []
        original_positions: list[numpy.ndarray] = []
        prep_stats_accum: dict[str, float] = {}
        microbatch_scratch_time = 0.0
        microbatch_metadata_time = 0.0
        microbatch_handoff_time = 0.0
        sampler_window_time = 0.0
        ensure_variants_time = 0.0
        scratch_slots = self._ensure_pipeline_microbatch_scratch(
            len(chunks),
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
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
        )

        handoff_offset = 0
        for chunk, scratch_slot in zip(chunks, scratch_slots, strict=True):
            scratch_start = time.time()
            (
                mb_scheduled,
                mb_active,
                mb_token_ids,
                mb_num_computed,
                mb_temperature,
                mb_top_p,
                mb_top_k,
                mb_min_p,
                mb_frequency,
                mb_presence,
                mb_repetition,
                mb_page_table,
            ) = self._populate_pipeline_microbatch_scratch(
                scratch_slot,
                chunk=chunk,
                scheduled_full_cpu=scheduled_full_cpu,
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
            )
            microbatch_scratch_time += time.time() - scratch_start

            metadata_start = time.time()
            batch_metadata, input_ids_buf, position_ids_buf, _scheduled_dev, _active_dev = self.prepare_batch_metadata(
                num_tokens_static=microbatch_num_tokens,
                scheduled_full_cpu=mb_scheduled,
                active_mask_full_cpu=mb_active,
                input_ids_buf=input_ids_buf,
                position_ids_buf=position_ids_buf,
                token_ids_cpu=mb_token_ids,
                num_computed_tokens_cpu=mb_num_computed,
                temperature_cpu=mb_temperature,
                top_p_cpu=mb_top_p,
                top_k_cpu=mb_top_k,
                min_p_cpu=mb_min_p,
                frequency_penalties_cpu=mb_frequency,
                presence_penalties_cpu=mb_presence,
                repetition_penalties_cpu=mb_repetition,
                page_table_cpu=mb_page_table,
                page_table_version=None,
                padded_num_reqs_in=microbatch_padded_reqs,
            )
            microbatch_metadata_time += time.time() - metadata_start
            local_count = int(chunk.shape[0])
            if device_token_handoff is not None and local_count > 0:
                handoff_start = time.time()
                handoff_offset_dev, handoff_count_dev = self._get_pipeline_handoff_scalars(
                    offset=handoff_offset,
                    count=local_count,
                )
                local_handoff = DeviceInputTokenHandoff(
                    input_positions=device_token_handoff.input_positions,
                    token_ids=device_token_handoff.token_ids,
                    count=handoff_count_dev,
                    offset=handoff_offset_dev,
                )
                batch_metadata = self._apply_device_token_handoff(batch_metadata, local_handoff)
                microbatch_handoff_time += time.time() - handoff_start
            handoff_offset += local_count
            for key, value in self._batch_preparer.last_prep_stats.items():
                prep_stats_accum[key] = prep_stats_accum.get(key, 0.0) + float(value)

            input_batches.append((self.graphstate, self.graphother, self.kv_pages, batch_metadata))
            metadata_batches.append(batch_metadata)
            original_positions.append(numpy.asarray(chunk, dtype=numpy.int32))

        sampler_window_start = time.time()
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
        sampler_window_time += time.time() - sampler_window_start
        gather_positions_cpu = self._sampler_gather_positions_cpu.copy()
        sampling_seeds_cpu = self._sampler_sampling_seeds_cpu.copy()
        scatter_positions_cpu = self._sampler_scatter_positions_cpu.copy()
        compact_window_row_indices_cpu = self._sampler_window_row_indices_cpu.copy()
        compact_scheduled_cpu = self._sampler_scheduled_cpu.copy()
        compact_seq_lens_cpu = self._sampler_seq_lens_cpu.copy()
        compact_active_mask_cpu = self._sampler_active_mask_cpu.copy()
        compact_temperature_cpu = self._sampler_temperature_cpu.copy()
        compact_top_p_cpu = self._sampler_top_p_cpu.copy()
        compact_top_k_cpu = self._sampler_top_k_cpu.copy()
        compact_min_p_cpu = self._sampler_min_p_cpu.copy()
        compact_frequency_penalties_cpu = self._sampler_frequency_penalties_cpu.copy()
        compact_presence_penalties_cpu = self._sampler_presence_penalties_cpu.copy()
        compact_repetition_penalties_cpu = self._sampler_repetition_penalties_cpu.copy()
        need_penalties = bool(
            numpy.any(frequency_penalties_cpu[:padded_num_reqs] != 0.0)
            or numpy.any(presence_penalties_cpu[:padded_num_reqs] != 0.0)
            or numpy.any(repetition_penalties_cpu[:padded_num_reqs] != 1.0)
        )
        ensure_start = time.time()
        self._ensure_runtime_variants(
            num_tokens=microbatch_num_tokens,
            padded_num_reqs=padded_num_reqs,
            sampler_padded_num_reqs=sampler_padded_num_reqs,
            compile_split_pp_path=True,
        )
        ensure_variants_time += time.time() - ensure_start

        prep_took = time.time() - start_prep

        start_backbone = time.time()
        with forward_autotune_only():
            backbone_outputs_list = self._model_executor.execute_backbones_many(
                num_tokens=microbatch_num_tokens,
                input_batches=input_batches,
            )
        self.kv_pages = backbone_outputs_list[-1].kv_pages
        backbone_took = time.time() - start_backbone

        start_combine = time.time()
        hidden_parts: list[jax.Array] = []
        logits_index_parts: list[jax.Array] = []
        position_parts: list[jax.Array] = []
        positions_are_prefix = bool(
            active_count <= int(padded_num_reqs)
            and active_positions.shape[0] == active_count
            and numpy.array_equal(active_positions, numpy.arange(active_count, dtype=active_positions.dtype))
        )
        for metadata, backbone_out, positions in zip(
            metadata_batches,
            backbone_outputs_list,
            original_positions,
            strict=True,
        ):
            local_count = int(positions.shape[0])
            if positions_are_prefix:
                logits_indices = self._get_pipeline_logits_indices(local_count, backbone_out.hidden_states)
            else:
                logits_indices = metadata.logits_indices[:local_count]
                logits_indices = replicate_on_array_mesh(logits_indices, backbone_out.hidden_states)
            hidden_parts.append(backbone_out.hidden_states)
            logits_index_parts.append(logits_indices)
            if not positions_are_prefix:
                scatter_idx = replicate_on_array_mesh(jnp.asarray(positions, dtype=jnp.int32), backbone_out.hidden_states)
                position_parts.append(scatter_idx)

        if positions_are_prefix:
            combined_hidden = None
        else:
            combined_hidden = _combine_pipeline_hidden_parts(
                tuple(hidden_parts),
                tuple(logits_index_parts),
                tuple(position_parts),
                padded_num_reqs=int(padded_num_reqs),
            )
        combine_took = time.time() - start_combine

        start_lm_head = time.time()
        if positions_are_prefix:
            lm_head_fn = self._model_executor.get_pipeline_lm_head(
                padded_num_reqs=padded_num_reqs,
                part_rows=tuple(int(positions.shape[0]) for positions in original_positions),
            )
            logits = lm_head_fn(self.graphstate, self.graphother, tuple(hidden_parts), tuple(logits_index_parts))
        else:
            combined_hidden = self._model_executor._place_lm_head_hidden(combined_hidden)
            lm_head_fn = self._model_executor.get_lm_head(padded_num_reqs=padded_num_reqs)
            logits = lm_head_fn(self.graphstate, self.graphother, combined_hidden)
        lm_head_took = time.time() - start_lm_head

        start_sampler = time.time()
        rng_key_out, out_tokens_full, valid_mask_full, token_counts_out = self.sample_tokens(
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            sampler_padded_num_reqs=sampler_padded_num_reqs,
            sampler_num_reqs=sampler_num_reqs,
            sampler_total_tokens=sampler_total_tokens,
                req_num_tokens_full_cpu=req_num_tokens_full_cpu,
            logits=logits,
            rng_key=self.rng_key,
            gather_positions_cpu=gather_positions_cpu,
            sampling_seeds_cpu=sampling_seeds_cpu,
            scatter_positions_cpu=scatter_positions_cpu,
            compact_window_row_indices_cpu=compact_window_row_indices_cpu,
            compact_scheduled_cpu=compact_scheduled_cpu,
            compact_seq_lens_cpu=compact_seq_lens_cpu,
            compact_active_mask_cpu=compact_active_mask_cpu,
            compact_temperature_cpu=compact_temperature_cpu,
            compact_top_p_cpu=compact_top_p_cpu,
            compact_top_k_cpu=compact_top_k_cpu,
            compact_min_p_cpu=compact_min_p_cpu,
            compact_frequency_penalties_cpu=compact_frequency_penalties_cpu,
            compact_presence_penalties_cpu=compact_presence_penalties_cpu,
            compact_repetition_penalties_cpu=compact_repetition_penalties_cpu,
            need_penalties=need_penalties,
        )
        self.rng_key = rng_key_out
        self._sampler_token_counts = token_counts_out
        sampler_enqueue_took = time.time() - start_sampler
        exec_took = backbone_took + combine_took + lm_head_took + sampler_enqueue_took

        if wait_for_outputs:
            jax.block_until_ready(logits)
            start_sample = time.time()
            jax.block_until_ready(out_tokens_full)
            sample_took = time.time() - start_sample
        else:
            sample_took = 0.0

        metrics = {
            "exec_time": exec_took,
            "sample_time": sample_took,
            "prep_time": prep_took,
            "execute_overhead_time": 0.0,
            "buckets_processed": int(microbatch_num_tokens) * len(chunks),
            "token_bucket": int(microbatch_num_tokens),
            "padded_num_reqs": int(microbatch_padded_reqs),
            "sampler_padded_num_reqs": int(sampler_padded_num_reqs),
            "sampler_num_reqs": int(active_count),
            "pp_microbatches": len(chunks),
            "pp_backbone_time": backbone_took,
            "pp_combine_time": combine_took,
            "pp_lm_head_time": lm_head_took,
            "pp_sampler_enqueue_time": sampler_enqueue_took,
            "pp_microbatch_scratch_time": microbatch_scratch_time,
            "pp_microbatch_metadata_time": microbatch_metadata_time,
            "pp_microbatch_handoff_time": microbatch_handoff_time,
            "pp_sampler_window_time": sampler_window_time,
            "pp_ensure_variants_time": ensure_variants_time,
        }
        metrics.update(self._model_executor.last_pipeline_stats())
        metrics.update(prep_stats_accum)

        return (
            out_tokens_full,
            valid_mask_full,
            input_ids_buf,
            position_ids_buf,
            backbone_outputs_list[-1].hidden_states,
            logits,
            metrics,
        )

    def execute_model(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        inputs: StepFunctionInputs,
    ) -> ModelStepOutputs:
        """Run the compiled model forward step and update self.kv_pages.

        Executes the pre-compiled model step function, computing hidden states
        and logits while updating the KV cache with new attention states.

        Args:
            num_tokens: Number of tokens for bucket selection.
            padded_num_reqs: Padded request count for bucket selection.
            inputs: Consolidated step function inputs containing kv_pages,
                batch_metadata, and other required tensors.

        Returns:
            ModelStepOutputs containing updated kv_pages, hidden_states, and
            logits.

        Note:
            This method updates self.kv_pages in-place with the new cache state.
            The returned outputs.kv_pages is the same reference. The method does
            not block on completion, allowing the caller to pipeline work like
            enqueuing sampling before synchronizing.
        """
        model_fn = self._model_executor.get_compiled(num_tokens=num_tokens, padded_num_reqs=padded_num_reqs)
        # Do not block here: allow the caller to pipeline dependent work
        # (e.g. enqueue sampling) before synchronizing.
        with forward_autotune_only():
            outputs = model_fn(self.graphstate, self.graphother, inputs.kv_pages, inputs.batch_metadata)
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

        No-op when the cache is not a :class:`HybridCache` (purely
        attention models have no recurrent state to clear) or when no
        slots were freed.

        Args:
            slot_indices: Row indices in ``[0, max_num_reqs)`` whose
                recurrent state should be zeroed. Empty list short-circuits.
        """
        if not slot_indices:
            return
        cache = self.kv_pages
        if not isinstance(cache, HybridCache):
            return

        changed = False
        new_views = list(cache.views)
        for idx, view in enumerate(new_views):
            rec = None
            if isinstance(view, RecurrentCacheView):
                rec = view
            elif isinstance(view, ParallelHybridCacheView) and view.recurrent is not None:
                rec = view.recurrent
            else:
                continue

            new_conv = rec.conv_state
            new_rec = rec.recurrent_state
            slot_arr = jnp.array(slot_indices, dtype=jnp.int32)
            # Build mask once — conv_state and recurrent_state share the batch dim.
            n_slots = new_conv.shape[0] if new_conv is not None else (new_rec.shape[0] if new_rec is not None else 0)
            if n_slots == 0:
                continue
            keep_mask = jnp.ones(n_slots, dtype=jnp.bool_).at[slot_arr].set(False)
            if new_conv is not None:
                new_conv = jnp.where(keep_mask.reshape(-1, *([1] * (new_conv.ndim - 1))), new_conv, 0)
            if new_rec is not None:
                new_rec = jnp.where(keep_mask.reshape(-1, *([1] * (new_rec.ndim - 1))), new_rec, 0)

            new_recurrent = rec.replace(conv_state=new_conv, recurrent_state=new_rec)
            if isinstance(view, ParallelHybridCacheView):
                new_views[idx] = view.replace(
                    recurrent=new_recurrent,
                    conv_state=new_conv,
                    recurrent_state=new_rec,
                )
            else:
                new_views[idx] = new_recurrent
            changed = True

        if changed:
            self.kv_pages = HybridCache(views=new_views)

    def sample_tokens(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        *,
        sampler_padded_num_reqs: int,
        sampler_num_reqs: int,
        sampler_total_tokens: int,
        req_num_tokens_full_cpu: numpy.ndarray,
        logits: jax.Array,
        rng_key: jax.Array,
        gather_positions_cpu: numpy.ndarray,
        sampling_seeds_cpu: numpy.ndarray,
        scatter_positions_cpu: numpy.ndarray,
        compact_window_row_indices_cpu: numpy.ndarray,
        compact_scheduled_cpu: numpy.ndarray,
        compact_seq_lens_cpu: numpy.ndarray,
        compact_active_mask_cpu: numpy.ndarray,
        compact_temperature_cpu: numpy.ndarray,
        compact_top_p_cpu: numpy.ndarray,
        compact_top_k_cpu: numpy.ndarray,
        compact_min_p_cpu: numpy.ndarray,
        compact_frequency_penalties_cpu: numpy.ndarray,
        compact_presence_penalties_cpu: numpy.ndarray,
        compact_repetition_penalties_cpu: numpy.ndarray,
        need_penalties: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Run the compiled sampler step over only the rows that need sampling.

        Executes the pre-compiled sampler function, converting model logits
        into sampled tokens based on per-request sampling parameters.

        Args:
            num_tokens: Number of tokens for bucket selection.
            padded_num_reqs: Model-side padded request count for the current window.
            sampler_padded_num_reqs: Sampler-side padded request count after
                compacting out zero-token rows.
            sampler_num_reqs: Actual number of compacted sampler rows.
            sampler_total_tokens: Total scheduled tokens in the compacted sampler batch.
            req_num_tokens_full_cpu: Target token count per request [max_num_reqs].
            logits: Model output logits [padded_num_reqs, vocab_size].
            rng_key: JAX random key for stochastic sampling.
            gather_positions_cpu: Model-row indices used to gather compact logits.
            sampling_seeds_cpu: Original model-row indices used to preserve
                exact per-row RNG fold-ins.
            scatter_positions_cpu: Output positions used to scatter compact
                results back to the model-window layout.
            compact_window_row_indices_cpu: Global request-row indices aligned
                with the compact sampler rows.
            compact_scheduled_cpu: Scheduled token counts for compact rows.
            compact_seq_lens_cpu: Sequence lengths after the forward pass.
            compact_active_mask_cpu: Active sampler rows.
            compact_temperature_cpu: Temperature per compact row.
            compact_top_p_cpu: Top-p per compact row.
            compact_top_k_cpu: Top-k per compact row.
            compact_min_p_cpu: Min-p per compact row.
            compact_frequency_penalties_cpu: Frequency penalty per compact row.
            compact_presence_penalties_cpu: Presence penalty per compact row.
            compact_repetition_penalties_cpu: Repetition penalty per compact row.
            need_penalties: Whether any request in the window uses penalties.

        Returns:
            Tuple of (updated_rng_key, sampled_tokens, valid_mask, updated_token_counts) where:
            - updated_rng_key: New RNG key for next step.
            - sampled_tokens: Generated token IDs [padded_num_reqs], -1 for invalid.
            - valid_mask: Boolean mask indicating valid samples [padded_num_reqs].
            - updated_token_counts: Exact device-side token counts [max_num_reqs, vocab_size].

        Note:
            This method does not block on completion, allowing the caller to
            overlap host work while the device executes. The caller should
            synchronize on the returned arrays when ready to use them.
        """
        sampler_fn = self._sampler_executor.get_compiled(
            num_tokens=num_tokens,
            padded_num_reqs=sampler_padded_num_reqs,
        )
        sampler_sharding = getattr(self, "_sampler_sharding", self._empty_sharding)
        if need_penalties:
            self._ensure_sampler_penalty_state()
        token_counts_full = (
            self._sampler_token_counts if self._sampler_penalty_state_ready else self._sampler_zero_token_counts
        )
        sampler_packed_i32 = self._sampler_packed_i32_cpu_by_reqs.get(int(sampler_padded_num_reqs))
        if sampler_packed_i32 is None:
            sampler_packed_i32 = numpy.zeros((7, int(sampler_padded_num_reqs)), dtype=numpy.int32)
            self._sampler_packed_i32_cpu_by_reqs[int(sampler_padded_num_reqs)] = sampler_packed_i32
        sampler_packed_f32 = self._sampler_packed_f32_cpu_by_reqs.get(int(sampler_padded_num_reqs))
        if sampler_packed_f32 is None:
            sampler_packed_f32 = numpy.zeros((6, int(sampler_padded_num_reqs)), dtype=numpy.float32)
            self._sampler_packed_f32_cpu_by_reqs[int(sampler_padded_num_reqs)] = sampler_packed_f32

        sampler_packed_f32[0] = compact_temperature_cpu[:sampler_padded_num_reqs]
        sampler_packed_f32[1] = compact_top_p_cpu[:sampler_padded_num_reqs]
        sampler_packed_f32[2] = compact_min_p_cpu[:sampler_padded_num_reqs]
        sampler_packed_f32[3] = compact_frequency_penalties_cpu[:sampler_padded_num_reqs]
        sampler_packed_f32[4] = compact_presence_penalties_cpu[:sampler_padded_num_reqs]
        sampler_packed_f32[5] = compact_repetition_penalties_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[0] = sampling_seeds_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[1] = compact_scheduled_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[2] = compact_seq_lens_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[3] = compact_window_row_indices_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[4] = compact_active_mask_cpu[:sampler_padded_num_reqs].astype(numpy.int32)
        sampler_packed_i32[5] = compact_top_k_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[6].fill(0)
        if int(sampler_num_reqs) > 0:
            live_gather_positions = gather_positions_cpu[: int(sampler_num_reqs)]
            sampler_packed_i32[6, : int(sampler_num_reqs)] = req_num_tokens_full_cpu[live_gather_positions]
        self._sampler_packed_misc_i32_cpu[0] = numpy.int32(sampler_num_reqs)
        self._sampler_packed_misc_i32_cpu[1] = numpy.int32(sampler_total_tokens)

        sampler_host_payload = (sampler_packed_f32, sampler_packed_i32, self._sampler_packed_misc_i32_cpu)
        if jax.process_count() > 1:
            sampler_host_payload = multihost_utils.broadcast_one_to_all(sampler_host_payload)
        with jax.named_scope("easydel/esurge/sampler/place_metadata"):
            packed_f32, packed_i32, packed_misc_i32 = _device_put_tree_uniform(sampler_host_payload, sampler_sharding)
        sampler_prefix = self._sampler_prefix_cpu_by_reqs.get(int(sampler_padded_num_reqs))
        if sampler_prefix is None:
            sampler_prefix = numpy.arange(sampler_padded_num_reqs, dtype=numpy.int32)
            self._sampler_prefix_cpu_by_reqs[int(sampler_padded_num_reqs)] = sampler_prefix
        prefix_gather_layout = numpy.array_equal(
            gather_positions_cpu[:sampler_padded_num_reqs],
            sampler_prefix,
        )
        prefix_scatter_layout = numpy.array_equal(
            scatter_positions_cpu[:sampler_padded_num_reqs],
            sampler_prefix,
        )
        with jax.named_scope("easydel/esurge/sampler/gather_logits"):
            if prefix_gather_layout:
                compact_logits = logits[:sampler_padded_num_reqs]
            else:
                logits_sharding = logits.sharding
                gather_positions = jax.device_put(gather_positions_cpu[:sampler_padded_num_reqs], sampler_sharding)
                gather_positions_for_logits = jax.device_put(gather_positions, logits_sharding)
                compact_logits = logits[gather_positions_for_logits]
            if getattr(compact_logits, "sharding", None) != sampler_sharding:
                compact_logits = jax.device_put(compact_logits, sampler_sharding)

        with jax.named_scope("easydel/esurge/sampler/run_jit"):
            rng_key, compact_tokens, compact_valid_mask, token_counts_full = sampler_fn(
                packed_f32,
                packed_i32,
                packed_misc_i32,
                compact_logits,
                rng_key,
                token_counts_full,
            )
        if prefix_scatter_layout:
            return rng_key, compact_tokens, compact_valid_mask, token_counts_full
        with jax.named_scope("easydel/esurge/sampler/scatter_to_model_rows"):
            scatter_positions = jax.device_put(scatter_positions_cpu[:sampler_padded_num_reqs], sampler_sharding)
            out_tokens_full, valid_mask_full = self._scatter_sampler_outputs(
                compact_tokens,
                compact_valid_mask,
                scatter_positions,
                padded_num_reqs=padded_num_reqs,
            )
        return rng_key, out_tokens_full, valid_mask_full, token_counts_full

    @staticmethod
    def _get_feasible_compile_pairs(
        num_tokens_paddings: list[int],
        reqs_padds: list[int],
    ) -> list[tuple[int, int]]:
        """Return only schedulable token/request bucket combinations.

        Every scheduled request consumes at least one token in a runner step,
        so a request bucket larger than the token bucket cannot be reached at
        runtime. Pruning those pairs shortens startup and avoids compiling dead
        executables such as `(32 tokens, 256 requests)`.
        """
        feasible_pairs: list[tuple[int, int]] = []
        for num_tokens in num_tokens_paddings:
            for reqs_padd in reqs_padds:
                if int(reqs_padd) > int(num_tokens):
                    continue
                feasible_pairs.append((int(num_tokens), int(reqs_padd)))
        return feasible_pairs

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
        """Compile model execution functions for various input configurations.

        Pre-compiles functions for different combinations of token counts and request
        counts to avoid runtime compilation overhead. This enables seamless switching
        between different batch sizes during inference.

        Args:
            num_tokens_paddings: List of token count configurations to compile.
            num_reqs_max_model_len: Maximum number of requests at max model length.
            max_pages_per_req: Maximum number of KV cache pages per request.
            max_num_reqs: Maximum number of concurrent requests.
            metadata: Pages cache metadata containing configuration details.
            prune_infeasible_pairs: Whether to skip compile pairs where the
                padded request bucket exceeds the token bucket. This is only
                safe when runtime compacts interior zero-token rows before
                bucket selection.

        Note:
            Compilation progress is logged using a progress bar. The total number
            of compilations is len(num_tokens_paddings) * number of unique padded
            request counts.

        Example:
            >>> executor.compile(
            ...     num_tokens_paddings=[128, 256, 512, 1024],
            ...     num_reqs_max_model_len=16,
            ...     max_pages_per_req=64,
            ...     max_num_reqs=32,
            ...     metadata=cache_metadata
            ... )
        """

        self._runtime_compile_metadata = metadata
        self._runtime_compile_max_num_reqs = int(max_num_reqs)

        if self.use_aot_forward:
            self.clear_cache()
        if num_reqs_paddings:
            reqs_padds = sorted({int(n) for n in num_reqs_paddings if 0 < int(n) <= max_num_reqs})
        else:
            ufn = partial(_get_padded_num_reqs_with_upper_limit, min_input_pad=self.min_input_pad)
            reqs_padds = sorted({ufn(n, max_num_reqs) for n in range(1, max_num_reqs + 1)})
        if not reqs_padds:
            reqs_padds = [max_num_reqs]
        sampler_ufn = partial(
            _get_padded_num_reqs_with_upper_limit,
            min_input_pad=int(self._sampler_min_input_pad),
        )
        sampler_reqs_padds = sorted(
            {
                *reqs_padds,
                *(sampler_ufn(n, max_num_reqs) for n in range(1, max_num_reqs + 1)),
            }
        )
        if prune_infeasible_pairs:
            sampler_compile_pairs = self._get_feasible_compile_pairs(num_tokens_paddings, sampler_reqs_padds)
        else:
            sampler_compile_pairs = [
                (int(num_tokens), int(reqs_padd))
                for num_tokens in num_tokens_paddings
                for reqs_padd in sampler_reqs_padds
            ]

        model_tokens = sorted({int(t) for t in num_tokens_paddings})
        self._model_num_tokens_paddings = model_tokens
        all_reqs_padds = sorted({int(r) for _, r in sampler_compile_pairs})
        sampler_reqs_padds_to_compile = all_reqs_padds
        if self.model.mesh.is_mpmd:
            if self._runtime_lazy_compile:
                logger.info(
                    "MPMD lazy compilation enabled: warming a minimal decode set; "
                    "additional buckets compile on first use."
                )
                model_tokens_to_compile = [model_tokens[0]]
                reqs_to_compile = [reqs_padds[0]]
                sampler_reqs_padds_to_compile = [
                    sampler_reqs_padds_to_compile[0]
                ] if sampler_reqs_padds_to_compile else reqs_to_compile
            else:
                model_tokens_to_compile = model_tokens
                reqs_to_compile = all_reqs_padds

            if self._model_executor.supports_pipeline_model_step:
                pp_model_reqs_to_compile = sorted({*reqs_to_compile, *sampler_reqs_padds_to_compile})
                pp_model_pairs = [
                    (int(num_tokens), int(reqs_padd))
                    for num_tokens in model_tokens_to_compile
                    for reqs_padd in pp_model_reqs_to_compile
                    if int(reqs_padd) <= int(num_tokens)
                ]
                pp_model_progress = ProgressLogger("eSurge-pp-model-step", logger)
                for idx, (num_tokens, reqs_padd) in enumerate(pp_model_pairs):
                    pp_model_progress.update(
                        idx,
                        len(pp_model_pairs),
                        f"Compiling [{idx + 1}/{len(pp_model_pairs)}]: {num_tokens:5d} tokens / {reqs_padd:2d} reqs",
                    )
                    self._compile_pipeline_model_step_variant(
                        num_tokens=num_tokens,
                        padded_num_reqs=reqs_padd,
                        max_num_reqs=max_num_reqs,
                        metadata=metadata,
                    )
                pp_model_progress.complete(
                    f"All {len(pp_model_pairs)} fused PP model-step compilations completed"
                )
            compile_split_pp_path = (not self._model_executor.supports_pipeline_model_step) or (
                not self._runtime_lazy_compile
            )
            if compile_split_pp_path:
                backbone_progress = ProgressLogger("eSurge-backbone", logger)
                for idx, num_tokens in enumerate(model_tokens_to_compile):
                    backbone_progress.update(
                        idx,
                        len(model_tokens_to_compile),
                        f"Compiling [{idx + 1}/{len(model_tokens_to_compile)}]: {num_tokens:5d} tokens",
                    )
                    self._compile_backbone_variant(
                        num_tokens=num_tokens,
                        max_num_reqs=max_num_reqs,
                        metadata=metadata,
                    )
                backbone_progress.complete(f"All {len(model_tokens_to_compile)} backbone compilations completed")

            lm_head_progress = ProgressLogger("eSurge-head-sampler", logger)
            lm_head_reqs_to_compile = reqs_to_compile if compile_split_pp_path else []
            total_phase2 = len(lm_head_reqs_to_compile) + len(sampler_reqs_padds_to_compile)
            phase2_idx = 0
            for reqs_padd in lm_head_reqs_to_compile:
                lm_head_progress.update(
                    phase2_idx,
                    total_phase2,
                    f"Compiling [{phase2_idx + 1}/{total_phase2}]: {reqs_padd:2d} padded requests",
                )
                self._compile_lm_head_variant(
                    padded_num_reqs=reqs_padd,
                    max_num_reqs=max_num_reqs,
                    metadata=metadata,
                )
                phase2_idx += 1
        else:
            reqs_to_compile = all_reqs_padds
            model_compile_pairs = self._get_feasible_compile_pairs(model_tokens, reqs_padds)
            if self._runtime_lazy_compile:
                logger.info(
                    "SPMD lazy compilation enabled: warming a minimal decode set; "
                    "additional buckets compile on first use."
                )
                model_compile_pairs = model_compile_pairs[:1]
                sampler_reqs_padds_to_compile = sampler_reqs_padds_to_compile[:1]
            model_step_progress = ProgressLogger("eSurge-model-step", logger)
            for idx, (num_tokens, reqs_padd) in enumerate(model_compile_pairs):
                model_step_progress.update(
                    idx,
                    len(model_compile_pairs),
                    f"Compiling [{idx + 1}/{len(model_compile_pairs)}]: {num_tokens:5d} tokens / {reqs_padd:2d} reqs",
                )
                self._compile_model_step_variant(
                    num_tokens=num_tokens,
                    padded_num_reqs=reqs_padd,
                    max_num_reqs=max_num_reqs,
                    metadata=metadata,
                )
            model_step_progress.complete(f"All {len(model_compile_pairs)} fused model-step compilations completed")

            lm_head_progress = ProgressLogger("eSurge-sampler", logger)
            total_phase2 = len(sampler_reqs_padds_to_compile)
            phase2_idx = 0
        sampler_dummy_tokens = max(model_tokens, default=max_num_reqs)
        for reqs_padd in sampler_reqs_padds_to_compile:
            num_tokens = max(int(sampler_dummy_tokens), int(reqs_padd))
            lm_head_progress.update(
                phase2_idx,
                total_phase2,
                f"Compiling [{phase2_idx + 1}/{total_phase2}]: sampler {reqs_padd:2d} padded requests",
            )
            self._compile_sampler_variant(
                num_tokens=num_tokens,
                max_num_reqs=max_num_reqs,
                padded_num_reqs=reqs_padd,
                metadata=metadata,
            )
            phase2_idx += 1
        if self.model.mesh.is_mpmd:
            lm_head_progress.complete(f"All {total_phase2} lm_head/sampler compilations completed")
        else:
            lm_head_progress.complete(f"All {total_phase2} sampler compilations completed")

    def _ensure_runtime_variants(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        sampler_padded_num_reqs: int,
        compile_split_pp_path: bool = False,
    ) -> None:
        """Compile missing buckets exactly when the scheduler reaches them.

        Full eager warmup is expensive on TPU: it asks XLA/SpectraX to
        materialize every token bucket before the first request, which can push
        large runs into long startup stalls or native runtime failures. The
        runtime already knows the exact bucket it is about to execute, so
        compile startup warms a small decode bucket and fills the remaining
        cache on first use.
        """
        if not self._runtime_lazy_compile:
            return
        metadata = self._runtime_compile_metadata
        max_num_reqs = self._runtime_compile_max_num_reqs
        if metadata is None or max_num_reqs is None:
            return

        with self._runtime_compile_lock:
            if not self.model.mesh.is_mpmd:
                if not self._model_executor.has_model_step(int(num_tokens), int(padded_num_reqs)):
                    logger.info(
                        "Lazy-compiling SPMD fused model-step bucket: %s tokens / %s requests",
                        int(num_tokens),
                        int(padded_num_reqs),
                    )
                    self._compile_model_step_variant(
                        num_tokens=int(num_tokens),
                        padded_num_reqs=int(padded_num_reqs),
                        max_num_reqs=max_num_reqs,
                        metadata=metadata,
                    )
            elif self._model_executor.supports_pipeline_model_step and not self._model_executor.has_model_step(
                int(num_tokens),
                int(padded_num_reqs),
            ):
                logger.info(
                    "Lazy-compiling MPMD fused model-step bucket: %s tokens / %s requests",
                    int(num_tokens),
                    int(padded_num_reqs),
                )
                self._compile_pipeline_model_step_variant(
                    num_tokens=int(num_tokens),
                    padded_num_reqs=int(padded_num_reqs),
                    max_num_reqs=max_num_reqs,
                    metadata=metadata,
                )
            need_split_pp_path = self.model.mesh.is_mpmd and (
                compile_split_pp_path or (not self._model_executor.supports_pipeline_model_step)
            )
            if need_split_pp_path and not self._model_executor.has_backbone(int(num_tokens)):
                logger.info("Lazy-compiling MPMD backbone bucket: %s tokens", int(num_tokens))
                self._compile_backbone_variant(
                    num_tokens=int(num_tokens),
                    max_num_reqs=max_num_reqs,
                    metadata=metadata,
                )
            if need_split_pp_path and not self._model_executor.has_lm_head(int(padded_num_reqs)):
                logger.info("Lazy-compiling MPMD lm_head bucket: %s requests", int(padded_num_reqs))
                self._compile_lm_head_variant(
                    padded_num_reqs=int(padded_num_reqs),
                    max_num_reqs=max_num_reqs,
                    metadata=metadata,
                )
            sampler_key = self._sampler_executor.cache_key(padded_num_reqs=int(sampler_padded_num_reqs))
            if not self._sampler_executor.has(sampler_key):
                logger.info("Lazy-compiling sampler bucket: %s requests", int(sampler_padded_num_reqs))
                self._compile_sampler_variant(
                    num_tokens=max(int(num_tokens), int(sampler_padded_num_reqs)),
                    max_num_reqs=max_num_reqs,
                    padded_num_reqs=int(sampler_padded_num_reqs),
                    metadata=metadata,
                )

    def _compile_model_step_variant(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
    ) -> None:
        """Compile the fused SPMD model step for one token/request bucket pair."""
        if self._model_executor.has_model_step(num_tokens, padded_num_reqs):
            return

        compargs = self.get_compile_configurations(
            self.kv_pages,
            self.rng_key,
            num_tokens,
            max_num_reqs,
            padded_num_reqs,
            metadata,
        )
        graphdef, graphstate, graphother, inputs = compargs
        model_out = self._model_executor.compile_model_step(
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            inputs=inputs,
        )
        if model_out is not None:
            self.kv_pages = model_out.kv_pages
        if self.use_aot_forward:
            warm_args = (graphstate, graphother, inputs)
            self._debug_baselines[f"{num_tokens}_{padded_num_reqs}_hash_in_model_step"] = _tree_hash(warm_args)

    def _compile_pipeline_model_step_variant(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
    ) -> None:
        """Compile the fused MPMD PP model step for one token/request bucket."""
        if not self._model_executor.supports_pipeline_model_step:
            return
        if self._model_executor.has_model_step(num_tokens, padded_num_reqs):
            return

        compargs = self.get_compile_configurations(
            self.kv_pages,
            self.rng_key,
            num_tokens,
            max_num_reqs,
            padded_num_reqs,
            metadata,
        )
        graphdef, graphstate, graphother, inputs = compargs
        model_out = self._model_executor.compile_pipeline_model_step(
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            inputs=inputs,
        )
        if model_out is not None:
            self.kv_pages = model_out.kv_pages
        if self.use_aot_forward:
            warm_args = (graphstate, graphother, inputs)
            self._debug_baselines[f"{num_tokens}_{padded_num_reqs}_hash_in_pp_model_step"] = _tree_hash(warm_args)

    def _compile_backbone_variant(
        self,
        *,
        num_tokens: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
    ) -> None:
        """Compile the backbone (transformer forward) for a token bucket.

        Keyed by ``num_tokens`` only.  Uses ``max_num_reqs`` as the dummy
        ``padded_num_reqs`` so that metadata shapes are fixed.
        """
        if self._model_executor.has_backbone(num_tokens):
            return

        compargs = self.get_compile_configurations(
            self.kv_pages,
            self.rng_key,
            num_tokens,
            max_num_reqs,
            max_num_reqs,  # padded_num_reqs = max_num_reqs (shapes are fixed)
            metadata,
        )
        graphdef, graphstate, graphother, inputs = compargs
        backbone_out = self._model_executor.compile_backbone(
            num_tokens=num_tokens,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            inputs=inputs,
        )
        if backbone_out is not None:
            self.kv_pages = backbone_out.kv_pages
        if self.use_aot_forward:
            warm_args = (graphstate, graphother, inputs)
            self._debug_baselines[f"{num_tokens}_hash_in_backbone"] = _tree_hash(warm_args)

    def _compile_lm_head_variant(
        self,
        *,
        padded_num_reqs: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
    ) -> None:
        """Compile the lm_head (gather + project) for a request bucket.

        Keyed by ``padded_num_reqs`` only.
        """
        if self._model_executor.has_lm_head(padded_num_reqs):
            return

        # We need graphdef/graphstate/graphother for the lm_head compilation.
        # num_tokens is irrelevant for lm_head — use max_num_reqs as a safe dummy.
        compargs = self.get_compile_configurations(
            self.kv_pages,
            self.rng_key,
            max_num_reqs,  # num_tokens (irrelevant for lm_head, just needs valid value)
            max_num_reqs,
            max_num_reqs,
            metadata,
        )
        graphdef, graphstate, graphother, inputs = compargs
        self._model_executor.compile_lm_head(
            padded_num_reqs=padded_num_reqs,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            inputs=inputs,
        )

    def _compile_sampler_variant(
        self,
        *,
        num_tokens: int,
        max_num_reqs: int,
        padded_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
        inputs: StepFunctionInputs | None = None,
    ) -> None:
        """Compile a sampler variant without requiring a matching model variant."""
        sampler_key = self._sampler_executor.cache_key(padded_num_reqs=padded_num_reqs)
        if self._sampler_executor.has(sampler_key):
            return
        if inputs is None:
            _, _, _, inputs = self.get_compile_configurations(
                self.kv_pages,
                self.rng_key,
                num_tokens,
                max_num_reqs,
                padded_num_reqs,
                metadata,
            )
        self._sampler_executor.compile(
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            inputs=inputs,
            metadata=inputs.batch_metadata,
        )
        if self.use_aot_forward:
            vocab_size = self.model.config.get_text_config().vocab_size
            dummy_logits = jnp.zeros(
                (padded_num_reqs, vocab_size),
                dtype=self.model.dtype,
                out_sharding=self._sampler_sharding,
            )
            sampler_args = (
                jnp.ones((6, padded_num_reqs), dtype=jnp.float32, out_sharding=self._sampler_sharding),
                jnp.zeros((7, padded_num_reqs), dtype=jnp.int32, out_sharding=self._sampler_sharding),
                jnp.array([padded_num_reqs, num_tokens], dtype=jnp.int32, out_sharding=self._sampler_sharding),
                dummy_logits,
                jax.device_put(inputs.rng_key, self._sampler_sharding),
                self._sampler_zero_token_counts,
            )
            self._debug_baselines[f"{num_tokens}_{padded_num_reqs}_hash_in_sampler"] = _tree_hash(sampler_args)

    def _compute_slot_mapping_v2(
        self,
        num_requests: int,
        scheduled: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        page_table_cpu: numpy.ndarray,
    ) -> tuple[numpy.ndarray, int]:
        """Compute slot mapping tensor for ragged-page attention v2.

        Delegates to the batch preparer's implementation to build the slot
        mapping that maps logical token positions to physical KV cache locations.

        Args:
            num_requests: Number of active requests.
            scheduled: Number of tokens scheduled per request.
            num_computed_tokens_cpu: Tokens already computed per request.
            page_table_cpu: Page table mapping request/page to physical page.

        Returns:
            Tuple of (slot_mapping, total_pages) where slot_mapping has shape
            [3, padded_num_slices] and total_pages is the number of pages
            touched by this batch.

        Note:
            This is used only for v2 attention (self._use_slot_mapping=True).
            For v3 attention, request_distribution is used instead.
        """
        return self._batch_preparer._compute_slot_mapping_v2(
            num_requests=num_requests,
            scheduled=scheduled,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            page_table_cpu=page_table_cpu,
        )

    def prepare_batch_metadata(
        self,
        num_tokens_static: int,
        scheduled_full_cpu: numpy.ndarray,  # CPU array instead of device
        active_mask_full_cpu: numpy.ndarray,  # CPU array instead of device
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        token_ids_cpu: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        temperature_cpu: numpy.ndarray,
        top_p_cpu: numpy.ndarray,
        top_k_cpu: numpy.ndarray,
        min_p_cpu: numpy.ndarray,
        frequency_penalties_cpu: numpy.ndarray,
        presence_penalties_cpu: numpy.ndarray,
        repetition_penalties_cpu: numpy.ndarray,
        page_table_cpu: numpy.ndarray,  # Pass page table as CPU array
        padded_num_reqs_in: int,
        page_table_version: int | None = None,
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
    ) -> tuple[BatchMetadata, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Prepare batch metadata using CPU-first computation.

        Delegates to the batch preparer to build all metadata on CPU and
        transfer to device in a single consolidated device_put call.

        Args:
            num_tokens_static: Static token count for bucket selection.
            scheduled_full_cpu: Tokens scheduled per request (CPU array).
            active_mask_full_cpu: Boolean mask for active requests (CPU array).
            input_ids_buf: Device buffer for input token IDs.
            position_ids_buf: Device buffer for position IDs.
            token_ids_cpu: All token IDs for all requests (CPU array).
            num_computed_tokens_cpu: Computed tokens per request (CPU array).
            temperature_cpu: Temperature per request (CPU array).
            top_p_cpu: Top-p per request (CPU array).
            top_k_cpu: Top-k per request (CPU array).
            min_p_cpu: Min-p per request (CPU array).
            frequency_penalties_cpu: Frequency penalty per request (CPU array).
            presence_penalties_cpu: Presence penalty per request (CPU array).
            repetition_penalties_cpu: Repetition penalty per request (CPU array).
            page_table_cpu: Page table (CPU array).
            padded_num_reqs_in: Requested padding for request count.
            page_table_version: Optional version for page table caching.
            mrope_position_ids_cpu: Optional mRoPE positions for VLMs.
            prefill_embeds_cpu: Optional prefill embeddings for VLMs.
            prefill_embeds_mask_cpu: Optional mask for prefill embeddings.
            visual_pos_masks_cpu: Optional visual position masks.
            deepstack_visual_embeds_cpu: Optional DeepStack visual embeddings.
            pixel_values: Optional raw image pixel values.
            image_grid_thw: Optional image grid shape.
            pixel_values_videos: Optional raw video pixel values.
            video_grid_thw: Optional video grid shape.

        Returns:
            Tuple of (batch_metadata, input_ids_buf, position_ids_buf,
            scheduled_full_dev, active_mask_full_dev) ready for model execution.
        """
        return self._batch_preparer.prepare_batch_metadata(
            num_tokens_static=num_tokens_static,
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
            padded_num_reqs_in=padded_num_reqs_in,
            mrope_position_ids_cpu=mrope_position_ids_cpu,
            prefill_embeds_cpu=prefill_embeds_cpu,
            prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
            visual_pos_masks_cpu=visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    def start_async_prep(
        self,
        num_tokens_static: int,
        scheduled_full_cpu: numpy.ndarray,
        active_mask_full_cpu: numpy.ndarray,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
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
        padded_num_reqs_in: int,
        page_table_version: int | None = None,
    ) -> None:
        """Start async device transfer for double-buffered batch preparation.

        Initiates an asynchronous device transfer for the next batch's metadata
        while the current batch is being processed.

        Args:
            num_tokens_static: Static token count for the next batch.
            scheduled_full_cpu: Tokens scheduled per request.
            active_mask_full_cpu: Active request mask.
            input_ids_buf: Device buffer for input IDs (unused, for API compat).
            position_ids_buf: Device buffer for positions (unused, for API compat).
            token_ids_cpu: Token IDs for all requests.
            num_computed_tokens_cpu: Computed tokens per request.
            temperature_cpu: Temperature per request.
            top_p_cpu: Top-p per request.
            top_k_cpu: Top-k per request.
            min_p_cpu: Min-p per request.
            frequency_penalties_cpu: Frequency penalty per request.
            presence_penalties_cpu: Presence penalty per request.
            repetition_penalties_cpu: Repetition penalty per request.
            page_table_cpu: Page table for all requests.
            padded_num_reqs_in: Requested padding for request count.
            page_table_version: Optional version for page table caching.

        Note:
            Call get_async_prep_result() to retrieve the prepared metadata.
        """
        self._batch_preparer.start_async_prep(
            num_tokens_static=num_tokens_static,
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
            padded_num_reqs_in=padded_num_reqs_in,
        )

    def get_async_prep_result(
        self,
    ) -> (
        tuple[
            tuple[BatchMetadata, jax.Array, jax.Array, jax.Array, jax.Array],
            dict,
        ]
        | None
    ):
        """Retrieve results from a previously started async batch preparation.

        Completes an async prep operation started by start_async_prep().

        Returns:
            If an async prep was pending, returns a tuple of:
            - (batch_metadata, input_ids_buf, position_ids_buf,
               scheduled_full_dev, active_mask_full_dev)
            - Metadata dict with timing and configuration information

            If no async prep was pending, returns None.
        """
        return self._batch_preparer.get_async_prep_result()

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

        # Create temporary buffer to generate dummy inputs
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
        uses_mrope_model = model_uses_mrope(self.model)

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

        # Get page table as CPU array
        page_table_cpu_dummy = temp_buffer.page_table[0].page_table_cpu

        (
            dummy_metadata,
            input_ids_buf,
            position_ids_buf,
            scheduled_full,
            active_mask_full,
        ) = self.prepare_batch_metadata(
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
