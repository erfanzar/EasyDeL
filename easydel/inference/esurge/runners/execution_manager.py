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
    - LRU cache with capacity of 64 compiled variants

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
import time
import typing
from functools import partial

import jax
import numpy
from eformer import escale as es
from eformer.jaximus import implicit
from eformer.loggings import ProgressLogger, get_logger
from eformer.pytree import key_path_to_str
from jax import numpy as jnp

from easydel.layers.caching import RaggedPagesCache, RaggedPagesCacheView

from ..utils import model_uses_mrope
from .execution_types import BatchMetadata, ModelStepOutputs, StepFunctionInputs
from .executors import BatchMetadataPreparer, ModelStepExecutor, SamplerExecutor
from .sequence_buffer import SequenceBuffer

DEBUG_MODE = False

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule

logger = get_logger("eSurge-ExecutionManager")

# Syncing inputs after host->device metadata transfer makes `prep_time` more accurate,
# but it adds a device round-trip that hurts throughput. Keep it opt-in.
SYNC_INPUTS_FOR_TIMING = bool(int(os.environ.get("EASURGE_SYNC_INPUTS_FOR_TIMING", "0")))


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


def _compute_sampling_valid_mask(
    *,
    i_reqs: jax.Array,
    num_requests: jax.Array,
    active_mask_slice: jax.Array,
    scheduled_slice: jax.Array,
    seq_lens_now: jax.Array,
    req_num_tokens_slice: jax.Array,
) -> jax.Array:
    """Compute which request slots are valid for sampling.

    A slot is valid if:
    - it is within the active request range (`i_reqs < num_requests`)
    - it is marked active (`active_mask_slice`)
    - it is scheduled (`scheduled_slice != 0`)
    - it has not finished (`seq_lens_now < req_num_tokens_slice`)
    """
    in_range = i_reqs < num_requests
    scheduled = scheduled_slice.astype(bool)
    not_finished = seq_lens_now < req_num_tokens_slice
    return in_range & active_mask_slice & scheduled & not_finished


def _device_put_tree_with_shardings(tree, shardings_tree):
    return jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s) if hasattr(x, "dtype") else x, tree, shardings_tree)


def _device_put_tree_uniform(tree, sharding):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    shardings_tree = jax.tree_util.tree_unflatten(treedef, [sharding] * len(leaves))
    return _device_put_tree_with_shardings(tree, shardings_tree)


def _tree_hash(tree):
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
        updated without recompilation. Compiled functions are cached in an LRU
        structure with 64-entry capacity.

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
        metadata: KV cache metadata (RaggedPagesCacheView).
        graphdef: Model graph definition (static structure).
        graphstate: Model graph state (weights, device-resident).
        graphother: Auxiliary model state (buffers, etc.).
        rng_key: JAX random key for sampling, threaded through steps.

    Private Attributes:
        _batch_preparer: CPU-first batch metadata builder and async transfer helper.
        _model_executor: Model-step executor with compiled-variant cache.
        _sampler_executor: Sampler executor with compiled-variant cache.
        _cache_capacity: Maximum cache entries (64).
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
        metadata: RaggedPagesCacheView = None,
        verbose: bool = False,
    ):
        """Initialize the executor manager.

        Args:
            model: The EasyDeL model instance.
            mesh: JAX sharding mesh for distributed execution.
            use_aot_forward: Whether to use Ahead-of-Time (AOT) compilation for model
                execution. When True (default), functions are pre-compiled for better
                performance. When False, uses Just-In-Time (JIT) compilation with
                the graph definition passed as a static argument.
            min_input_pad: Minimum padding for inputs.
            max_model_len: Maximum model sequence length.
            max_num_reqs: Maximum number of requests.
            max_num_tokens: Maximum number of tokens for batching.
            metadata: Pages cache metadata.
        """
        logger.info(f"initializing eSurge-ExecutionManager Version {metadata.version}")
        self.model = model
        self.mesh = model.mesh
        self.kv_pages = model.init_ragged_pages(metadata)
        self.use_aot_forward = use_aot_forward
        self.min_input_pad = min_input_pad
        self.max_model_len = max_model_len
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens if max_num_tokens is not None else max_model_len
        self.metadata = metadata
        self._metadata_version = metadata.version
        self._use_slot_mapping = metadata.version == "v2"
        self._use_request_distribution = not self._use_slot_mapping
        self.graphdef, self.graphstate, self.graphother = model.split_module()

        self.log_it = logger.info if verbose else logger.debug
        self._verbose = verbose

        self._empty_sharding = jax.NamedSharding(model.mesh, jax.sharding.PartitionSpec())

        self.rng_key = jax.device_put(jax.random.PRNGKey(0), self._empty_sharding)

        self._cache_capacity = 64
        self._debug_baselines = {}

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
            cache_capacity=self._cache_capacity,
            maybe_implicit=self.maybe_implicit,
        )
        self._sampler_executor = SamplerExecutor(
            model=self.model,
            max_model_len=self.max_model_len,
            empty_sharding=self._empty_sharding,
            use_aot_forward=self.use_aot_forward,
            cache_capacity=self._cache_capacity,
            maybe_implicit=self.maybe_implicit,
        )

    def clear_cache(self):
        self._model_executor.clear_cache()
        self._sampler_executor.clear_cache()
        self._debug_baselines.clear()

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
            shardings = es.extract_shardings(self.graphstate, self.mesh)
            self.graphstate = _device_put_tree_with_shardings(graphstate, shardings)

        if graphother is not None:
            shardings = es.extract_shardings(self.graphother, self.mesh)
            self.graphother = _device_put_tree_with_shardings(graphother, shardings)

        # Clear cached baselines so future diagnostics re-hash with new weights.
        self._debug_baselines.clear()

    def execute(
        self,
        num_tokens: int,
        scheduled_full_cpu: numpy.ndarray,  # CPU array
        req_num_tokens_full: jax.Array,
        active_mask_full_cpu: numpy.ndarray,  # CPU array
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        padded_num_reqs: int,
        token_ids_cpu: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        temperature_cpu: numpy.ndarray,
        top_p_cpu: numpy.ndarray,
        top_k_cpu: numpy.ndarray,
        min_p_cpu: numpy.ndarray,
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
            req_num_tokens_full: Target token count per request [max_num_reqs].
                Used to determine when requests have generated enough tokens.
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
            in-place on self after execution completes.

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
        start_prep = time.time()
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

        inputs = StepFunctionInputs(
            kv_pages=self.kv_pages,
            scheduled_full=scheduled_full,
            req_num_tokens_full=req_num_tokens_full,
            active_mask_full=active_mask_full,
            rng_key=self.rng_key,
            batch_metadata=batch_metadata,
        )
        # Syncing inputs here improves `prep_time` accuracy but adds a device
        # round-trip; keep it behind an explicit env flag.
        if self._verbose and SYNC_INPUTS_FOR_TIMING:
            inputs = jax.block_until_ready(inputs)
        prep_took = time.time() - start_prep
        if DEBUG_MODE:
            model_hash = _tree_hash((self.graphstate, self.graphother, inputs))
            model_hash_baseline = self._debug_baselines[f"{num_tokens}_{padded_num_reqs}_hash_in_model"]
            _tree_hash_diff(model_hash_baseline, model_hash)

        start_exec = time.time()
        model_outputs = self.execute_model(num_tokens=num_tokens, padded_num_reqs=padded_num_reqs, inputs=inputs)

        sampler_inputs = (
            batch_metadata,
            req_num_tokens_full,
            active_mask_full,
            model_outputs.logits,
            self.rng_key,
        )

        if DEBUG_MODE:
            sampler_hash = _tree_hash(sampler_inputs)
            sampler_hash_baseline = self._debug_baselines[f"{num_tokens}_{padded_num_reqs}_hash_in_sampler"]
            _tree_hash_diff(sampler_hash_baseline, sampler_hash)

        # Enqueue sampling immediately (it will run after the forward pass),
        # then synchronize on logits to measure forward time without an extra
        # host-side dispatch gap between the two computations.
        sampler_out = self.sample_tokens(
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            batch_metadata=batch_metadata,
            req_num_tokens_full=req_num_tokens_full,
            active_mask_full=active_mask_full,
            logits=model_outputs.logits,
            rng_key=self.rng_key,
        )
        jax.block_until_ready(model_outputs.logits)
        exec_took = time.time() - start_exec

        start_sample = time.time()
        rng_key_out, out_tokens_full, valid_mask_full = sampler_out
        jax.block_until_ready(out_tokens_full)
        self.rng_key = rng_key_out
        sample_took = time.time() - start_sample
        execute_total_took = time.time() - start_prep
        execute_overhead_took = execute_total_took - (prep_took + exec_took + sample_took)
        execute_overhead_took = max(0.0, float(execute_overhead_took))
        buckets_processed = batch_metadata.input_ids_buf.shape[-1]
        metrics = {
            "exec_time": exec_took,
            "sample_time": sample_took,
            "prep_time": prep_took,
            "execute_overhead_time": execute_overhead_took,
            "buckets_processed": buckets_processed,
            "token_bucket": int(num_tokens),
            "padded_num_reqs": int(padded_num_reqs),
        }
        try:
            metrics.update(getattr(self._batch_preparer, "last_prep_stats", {}) or {})
        except Exception:
            pass

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

    def execute_model(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        inputs: StepFunctionInputs,
    ) -> ModelStepOutputs:
        """Run the compiled model forward step and update `self.kv_pages`."""

        model_fn = self._model_executor.get_compiled(num_tokens=num_tokens, padded_num_reqs=padded_num_reqs)
        # Do not block here: allow the caller to pipeline dependent work
        # (e.g. enqueue sampling) before synchronizing.
        outputs = model_fn(self.graphstate, self.graphother, inputs.kv_pages, inputs.batch_metadata)
        self.kv_pages = outputs.kv_pages
        return outputs

    def sample_tokens(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        *,
        batch_metadata: BatchMetadata,
        req_num_tokens_full: jax.Array,
        active_mask_full: jax.Array,
        logits: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Run the compiled sampler step (no KV-cache mutation)."""

        sampler_fn = self._sampler_executor.get_compiled(num_tokens=num_tokens, padded_num_reqs=padded_num_reqs)
        # Keep this non-blocking so the caller can overlap host work while the
        # device enqueues sampling behind the forward pass.
        return sampler_fn(batch_metadata, req_num_tokens_full, active_mask_full, logits, rng_key)

    def compile(
        self,
        num_tokens_paddings: list[int],
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheView,
        num_reqs_paddings: list[int] | None = None,
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

        if self.use_aot_forward:
            self.clear_cache()
        if num_reqs_paddings:
            reqs_padds = sorted({int(n) for n in num_reqs_paddings if 0 < int(n) <= max_num_reqs})
        else:
            ufn = partial(_get_padded_num_reqs_with_upper_limit, min_input_pad=self.min_input_pad)
            reqs_padds = sorted({ufn(n, max_num_reqs) for n in range(1, max_num_reqs + 1)})
        if not reqs_padds:
            reqs_padds = [max_num_reqs]
        total_compilations = len(num_tokens_paddings) * len(reqs_padds)
        compilation_count = 0
        progress = ProgressLogger("eSurge", logger)
        for num_tokens in num_tokens_paddings:
            for reqs_padd in reqs_padds:
                progress.update(
                    compilation_count,
                    total_compilations,
                    f"Compiling [{compilation_count + 1}/{total_compilations}]: {num_tokens:5d} tokens, "
                    f"{reqs_padd:2d} padded requests",
                )
                self._step_compile(
                    num_tokens=num_tokens,
                    num_reqs_max_model_len=num_reqs_max_model_len,
                    max_pages_per_req=max_pages_per_req,
                    max_num_reqs=max_num_reqs,
                    padded_num_reqs=reqs_padd,
                    metadata=metadata,
                )
                compilation_count += 1
        progress.complete(f"All {total_compilations} compilations completed")

    def _step_compile(
        self,
        num_tokens: int,
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
        max_num_reqs: int,
        padded_num_reqs: int,
        metadata: RaggedPagesCacheView,
    ) -> None:
        """Compile a single step configuration.

        Internal method that compiles functions for a specific combination of
        token count and padded request count.

        Args:
            num_tokens: Number of tokens in this configuration.
            num_reqs_max_model_len: Maximum number of requests at max model length.
            max_pages_per_req: Maximum number of pages per request.
            max_num_reqs: Maximum number of requests.
            padded_num_reqs: Padded number of requests for this configuration.
            metadata: Pages cache metadata.

        Note:
            This method is called internally by compile() for each configuration.
        """
        compargs = self.get_compile_configurations(
            self.kv_pages,
            self.rng_key,
            num_tokens,
            max_num_reqs,
            padded_num_reqs,
            metadata,
        )
        graphdef, graphstate, graphother, inputs = compargs

        mode = "aot" if self.use_aot_forward else "jit"
        model_key = (num_tokens, padded_num_reqs, "model", mode)
        if not self._model_executor.has(model_key):
            model_out = self._model_executor.compile(
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
                self._debug_baselines[f"{num_tokens}_{padded_num_reqs}_hash_in_model"] = _tree_hash(warm_args)

        sampler_key = (num_tokens, padded_num_reqs, "sampler", mode)
        if not self._sampler_executor.has(sampler_key):
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
                    out_sharding=self._empty_sharding,
                )
                sampler_args = (
                    inputs.batch_metadata,
                    inputs.req_num_tokens_full,
                    inputs.active_mask_full,
                    dummy_logits,
                    inputs.rng_key,
                )
                self._debug_baselines[f"{num_tokens}_{padded_num_reqs}_hash_in_sampler"] = _tree_hash(sampler_args)

    def _compute_slot_mapping_v2(
        self,
        num_requests: int,
        scheduled: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        page_table_cpu: numpy.ndarray,
    ) -> tuple[numpy.ndarray, int]:
        """Rebuild slot_mapping tensor for ragged-page attention v2."""
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
        page_table_cpu: numpy.ndarray,
        padded_num_reqs_in: int,
        page_table_version: int | None = None,
    ) -> None:
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
        return self._batch_preparer.get_async_prep_result()

    def get_compiled_key(self, num_tokens: int, padded_num_reqs: int):
        """Retrieve pre-compiled model step function for given input dimensions.

        Args:
            num_tokens: Number of tokens in the input batch.
            padded_num_reqs: Padded number of requests for batching.

        Returns:
            Compiled fused step function for the specified number of tokens.
        """

        mode = "aot" if self.use_aot_forward else "jit"
        model_key = (num_tokens, padded_num_reqs, "model", mode)
        sampler_key = (num_tokens, padded_num_reqs, "sampler", mode)
        if self._model_executor.has(model_key):
            logger.debug(f"[CACHE HIT] model_key={model_key}")
        else:
            logger.warning(f"[CACHE MISS] key={model_key}! Will trigger recompilation (model)")
            logger.warning(f"Available keys in cache: {self._model_executor.cache_keys()}")
        if self._sampler_executor.has(sampler_key):
            logger.debug(f"[CACHE HIT] sampler_key={sampler_key}")
        else:
            logger.warning(f"[CACHE MISS] key={sampler_key}! Will trigger recompilation (sampler)")
            logger.warning(f"Available keys in cache: {self._sampler_executor.cache_keys()}")
        return (
            self._model_executor.get_compiled(num_tokens=num_tokens, padded_num_reqs=padded_num_reqs),
            self._sampler_executor.get_compiled(num_tokens=num_tokens, padded_num_reqs=padded_num_reqs),
        )

    def get_compile_configurations(
        self,
        kv_pages: RaggedPagesCache,
        rng_key: jax.random.PRNGKey,
        num_tokens: int,
        max_num_reqs: int,
        padded_num_reqs: int,
        metadata: RaggedPagesCacheView,
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

    @property
    def maybe_implicit(self):
        def no_implicit(fn):
            return fn

        if self.model.is_quantized:
            return implicit

        return no_implicit
