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
import time
import typing
from collections import OrderedDict
from functools import partial

import jax
import numpy
from eformer import escale as es
from eformer.loggings import ProgressLogger, get_logger
from eformer.pytree import key_path_to_str
from flax import nnx as nn
from jax import numpy as jnp
from jax._src import pjit

from easydel.layers.caching import RaggedPagesCache, RaggedPagesCacheView, RaggedPagesMetadata
from easydel.utils import ejit

from ..core.sampler import sample_tokens
from ..core.sampling_metadata import SamplingMetadata
from ..page_table import PAGE_TABLE_PADDING_VAL
from .execution_types import BatchMetadata, ModelStepOutputs, StepFunctionInputs
from .sequence_buffer import DeviceSequenceState, SequenceBuffer

DEBUG_MODE = False

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule

logger = get_logger("eSurge-ExecutionManager")


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


def _device_put_tree_with_shardings(tree, shardings_tree):
    return jax.tree_util.tree_map(
        lambda x, s: jax.device_put(x, s) if hasattr(x, "dtype") else x,
        tree,
        shardings_tree,
    )


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
        _model_step_fn: Model-only forward function (ejit-decorated).
        _sampling_fn: Sampler/update function (ejit-decorated).
        _model_lowerd_history: OrderedDict LRU cache of compiled model functions.
        _sampler_lowerd_history: OrderedDict cache for compiled sampler function.
        _cache_capacity: Maximum cache entries (64).
        _debug_baselines: Hash baselines for debugging recompilations.
        _empty_sharding: Default sharding (replicated across mesh).

    Example:
        >>> # Initialize manager
        >>> executor = ExecutionManager(
        ...     model=model,
        ...     mesh=mesh,
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
        mesh: jax.sharding.Mesh,
        kv_pages: RaggedPagesCache,
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
            kv_pages: Pages cache for KV cache management.
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
        logger.info("Initializing ExecutionManager")
        self.model = model
        self.mesh = mesh
        self.kv_pages = kv_pages
        self.use_aot_forward = use_aot_forward
        self.min_input_pad = min_input_pad
        self.max_model_len = max_model_len
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens if max_num_tokens is not None else max_model_len
        self.metadata = metadata
        self.graphdef, self.graphstate, self.graphother = model.split_module()

        self.log_it = logger.info if verbose else logger.debug

        self._empty_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())

        self.rng_key = jax.device_put(jax.random.PRNGKey(0), self._empty_sharding)

        self._model_step_fn: None | pjit.JitWrapped = None
        self._sampling_fn = self.get_sampling_fn()
        self._sampling_impl = self._sampling_fn
        self._cache_capacity = 64
        self._model_lowerd_history = OrderedDict()
        self._sampler_lowerd_history = OrderedDict()
        self._debug_baselines = {}

        # Pre-allocate CPU buffers for fast batch metadata preparation
        self._input_ids_cpu = numpy.zeros((max_num_tokens,), dtype=numpy.int32)
        self._positions_cpu = numpy.zeros((max_num_tokens,), dtype=numpy.int32)
        self._query_start_loc_cpu = numpy.zeros((max_num_reqs + 1,), dtype=numpy.int32)
        self._seq_lens_cpu = numpy.zeros((max_num_reqs,), dtype=numpy.int32)
        self._logits_indices_cpu = numpy.zeros((max_num_reqs,), dtype=numpy.int32)
        self._scheduled_cpu = numpy.zeros((max_num_reqs,), dtype=numpy.int32)
        self._arange_cpu = numpy.arange(max_num_tokens, dtype=numpy.int32)

        self.init_fns()

    def _model_cache_put(self, key, value):
        self._model_lowerd_history[key] = value
        self._model_lowerd_history.move_to_end(key)
        if len(self._model_lowerd_history) > self._cache_capacity:
            self._model_lowerd_history.popitem(last=False)

    def _model_cache_get(self, key):
        value = self._model_lowerd_history[key]
        self._model_lowerd_history.move_to_end(key)
        return value

    def _sampler_cache_put(self, key, value):
        self._sampler_lowerd_history[key] = value
        self._sampler_lowerd_history.move_to_end(key)
        if len(self._sampler_lowerd_history) > self._cache_capacity:
            self._sampler_lowerd_history.popitem(last=False)

    def _sampler_cache_get(self, key):
        value = self._sampler_lowerd_history[key]
        self._sampler_lowerd_history.move_to_end(key)
        return value

    def clear_cache(self):
        self._model_lowerd_history.clear()
        self._sampler_lowerd_history.clear()

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
            logger.info("Updating ExecutionManager graphs from provided model instance")
            self.model = model
            new_graphdef, new_graphstate, new_graphother = model.split_module()
            graphdef = new_graphdef if graphdef is None else graphdef
            graphstate = new_graphstate if graphstate is None else graphstate
            graphother = new_graphother if graphother is None else graphother

        if graphdef is None and graphstate is None and graphother is None:
            raise ValueError("No graph components supplied for update")

        if graphdef is not None:
            self.graphdef = graphdef

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
        device_state: DeviceSequenceState,
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
    ) -> tuple[
        DeviceSequenceState,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        """Execute a single fused inference step.

        Runs a pre-compiled fused function that combines input preparation, model
        forward pass, token sampling, and state updates in a single device dispatch.

        Args:
            num_tokens: Total tokens to process across all requests in this step.
                Must match a value from num_tokens_paddings used during compile().
            device_state: Current device-side sequence state (DeviceSequenceState).
                Contains token buffers, position tracking, and sampling parameters.
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
            Tuple of 10 elements:
                - device_state: Updated sequence state with new tokens written.
                - out_tokens_full: Generated tokens [max_num_reqs], -1 for invalid.
                - valid_mask_full: Boolean mask for valid generations [max_num_reqs].
                - input_ids_buf: Updated input buffer (may contain new tokens).
                - position_ids_buf: Updated position buffer.
                - query_start_loc_buf: Query start locations [max_num_reqs+1].
                - seq_lens_buf: Sequence lengths [max_num_reqs].
                - pages_tables_buf: Page tables [num_reqs, max_pages].
                - hidden_states: Last layer hidden states [num_tokens, hidden_dim].
                - logits: Output logits [padded_num_reqs, vocab_size].

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
        model_fn, sampler_fn = self.get_compiled_key(num_tokens, padded_num_reqs)
        start_prep = time.time()
        batch_metadata, input_ids_buf, position_ids_buf = self.prepare_batch_metadata(
            num_tokens_static=num_tokens,
            device_state=device_state,
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
        )
        prep_took = time.time() - start_prep

        # Convert CPU arrays to device for model execution
        scheduled_full = jax.device_put(jnp.asarray(scheduled_full_cpu), self._empty_sharding)
        active_mask_full = jax.device_put(jnp.asarray(active_mask_full_cpu), self._empty_sharding)

        inputs = StepFunctionInputs(
            device_state=device_state,
            kv_pages=self.kv_pages,
            scheduled_full=scheduled_full,
            req_num_tokens_full=req_num_tokens_full,
            active_mask_full=active_mask_full,
            rng_key=self.rng_key,
            batch_metadata=batch_metadata,
        )
        if DEBUG_MODE:
            model_hash = _tree_hash((self.graphstate, self.graphother, inputs))
            model_hash_baseline = self._debug_baselines[f"{num_tokens}_hash_in_model"]
            _tree_hash_diff(model_hash_baseline, model_hash)

        start_exec = time.time()
        model_outputs = model_fn(self.graphstate, self.graphother, inputs)
        exec_took = time.time() - start_exec

        self.kv_pages = model_outputs.kv_pages

        sampler_inputs = (
            batch_metadata,
            device_state,
            req_num_tokens_full,
            active_mask_full,
            model_outputs.logits,
            self.rng_key,
        )

        if DEBUG_MODE:
            sampler_hash = _tree_hash(sampler_inputs)
            sampler_hash_baseline = self._debug_baselines[f"{num_tokens}_hash_in_sampler"]
            _tree_hash_diff(sampler_hash_baseline, sampler_hash)

        start_sample = time.time()
        device_state, self.rng_key, out_tokens_full, valid_mask_full = sampler_fn(*sampler_inputs)
        sample_took = time.time() - start_sample

        self.log_it(f"model={exec_took} sampler={sample_took} prep={prep_took}")

        query_start_loc_buf = batch_metadata.query_start_loc
        seq_lens_buf = batch_metadata.seq_lens
        pages_tables_buf = batch_metadata.pages_tables
        hidden_states = model_outputs.hidden_states
        logits = model_outputs.logits

        return (
            device_state,
            out_tokens_full,
            valid_mask_full,
            input_ids_buf,
            position_ids_buf,
            query_start_loc_buf,
            seq_lens_buf,
            pages_tables_buf,
            hidden_states,
            logits,
        )

    def compile(
        self,
        num_tokens_paddings: list[int],
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheView,
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
        ufn = partial(_get_padded_num_reqs_with_upper_limit, min_input_pad=self.min_input_pad)
        reqs_padds = sorted({ufn(n, max_num_reqs) for n in range(1, max_num_reqs + 1)})
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
            num_reqs_max_model_len,
            max_pages_per_req,
            max_num_reqs,
            padded_num_reqs,
            metadata,
        )
        # compargs already contains properly prepared metadata from get_compile_configurations
        inputs = compargs[3]
        self._compile_model_step(num_tokens, compargs)
        self._compile_sampler(num_tokens, inputs, inputs.batch_metadata)

    def init_fns(self) -> None:
        """Initialize the fused step execution function.

        Initializes the model-only execution function. Sampling/state updates are
        handled by a separate ejit generated during initialization.

        Note:
            Called automatically during initialization. Should not be called
            directly by users.
        """
        self._model_step_fn = self.get_model_step_fn()

    def prepare_batch_metadata(
        self,
        num_tokens_static: int,
        device_state: DeviceSequenceState,
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
    ) -> tuple[BatchMetadata, jax.Array, jax.Array]:
        """Precompute batch metadata using CPU-first approach.

        Performs all metadata computation on CPU using NumPy for speed,
        then transfers to device in a single operation.

        Args:
            num_tokens_static: Number of tokens to process
            device_state: Device sequence state (for page tables and sampling params)
            scheduled_full: Tokens scheduled per request
            active_mask_full: Active request mask
            input_ids_buf: Input buffer (will be replaced)
            position_ids_buf: Position buffer (will be replaced)
            token_ids_cpu: NumPy array of token IDs [max_num_reqs, max_model_len]
            num_computed_tokens_cpu: NumPy array of computed tokens [max_num_reqs]

        Returns:
            Tuple of (BatchMetadata, input_ids_buf, position_ids_buf)
        """

        max_num_reqs = int(self.max_num_reqs)
        num_reqs_max_model_len = min(int(self.metadata.get_max_num_seqs()), max_num_reqs)

        # ========== NO DEVICE TRANSFERS! Everything on CPU ==========
        # scheduled_full_cpu and active_mask_full_cpu are already CPU arrays from scheduler
        scheduled_cpu = scheduled_full_cpu
        active_mask_cpu = active_mask_full_cpu

        # ========== All metadata computation on CPU using NumPy (FAST!) ==========

        # Compute num_requests
        num_requests = min(int(numpy.sum(active_mask_cpu)), max_num_reqs)
        mask_reqs = numpy.arange(max_num_reqs) < num_requests
        self._scheduled_cpu[:] = numpy.where(mask_reqs, scheduled_cpu, 0)
        scheduled = self._scheduled_cpu

        # Cumsum on CPU (much faster than device!)
        self._query_start_loc_cpu[0] = 0
        numpy.cumsum(scheduled[:num_requests], out=self._query_start_loc_cpu[1 : num_requests + 1])
        self._query_start_loc_cpu[num_requests + 1 :] = self._query_start_loc_cpu[num_requests]

        # Token gathering
        # Get request indices: [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = numpy.repeat(numpy.arange(num_requests, dtype=numpy.int32), scheduled[:num_requests])

        # Get batched arange: [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = numpy.concatenate([self._arange_cpu[:n] for n in scheduled[:num_requests]])

        # Actual number of tokens being processed (may be less than padded num_tokens_static)
        actual_num_tokens = len(req_indices)

        # Compute positions on CPU (use actual size for computation)
        positions_np = self._positions_cpu[:actual_num_tokens]
        numpy.add(num_computed_tokens_cpu[req_indices], arange, out=positions_np)

        # Token gathering on CPU
        token_indices = positions_np + req_indices * token_ids_cpu.shape[1]  # max_model_len
        numpy.take(token_ids_cpu.ravel(), token_indices, out=self._input_ids_cpu[:actual_num_tokens])

        # Pad remaining positions with zeros if actual < static
        if actual_num_tokens < num_tokens_static:
            self._input_ids_cpu[actual_num_tokens:num_tokens_static] = 0
            self._positions_cpu[actual_num_tokens:num_tokens_static] = 0

        # Sequence lengths on CPU
        self._seq_lens_cpu[:num_requests] = num_computed_tokens_cpu[:num_requests] + scheduled[:num_requests]
        self._seq_lens_cpu[num_requests:] = 0

        # Logits indices on CPU
        self._logits_indices_cpu[:num_requests] = self._query_start_loc_cpu[1 : num_requests + 1] - 1

        # Compute padded_num_reqs on CPU
        nr_safe = max(num_requests, 1)
        next_pow2 = 1 << (nr_safe - 1).bit_length()
        padded_num_reqs = min(self.min_input_pad if num_requests <= self.min_input_pad else next_pow2, max_num_reqs)

        # Page table already on CPU
        pt_src = page_table_cpu[: min(page_table_cpu.shape[0], num_reqs_max_model_len), :]
        mask_rows = numpy.arange(num_reqs_max_model_len) < min(num_requests, num_reqs_max_model_len)
        pages_tables_cpu = numpy.where(mask_rows[:, None], pt_src, PAGE_TABLE_PADDING_VAL)

        # Request distribution on CPU
        active_num_computed = numpy.where(mask_reqs, num_computed_tokens_cpu, 0)
        is_decode = (scheduled == 1) & (active_num_computed > 0)
        is_chunked_prefill = (scheduled > 1) & (active_num_computed > 0)
        decode_count = int(numpy.sum(is_decode))
        chunked_prefill_count = int(numpy.sum(is_chunked_prefill))
        boundary = min(decode_count + chunked_prefill_count, num_requests)
        request_distribution = numpy.array([decode_count, boundary, num_requests], dtype=numpy.int32)

        # Sampling params are already on CPU (passed as arguments), no device transfer needed!

        # ========== STEP 3: Single device transfer (ONE SHOT!) ==========

        # Transfer all CPU-computed metadata to device in one call
        input_ids_buf = jax.device_put(self._input_ids_cpu[: self.max_num_tokens], self._empty_sharding)
        position_ids_buf = jax.device_put(self._positions_cpu[: self.max_num_tokens], self._empty_sharding)

        # Transfer all metadata arrays
        qsl, seq_lens, logits_indices, pt, req_dist = jax.device_put(
            (
                self._query_start_loc_cpu,
                self._seq_lens_cpu,
                self._logits_indices_cpu,
                pages_tables_cpu,
                request_distribution,
            ),
            self._empty_sharding,
        )
        # Build BatchMetadata with device arrays
        metadata = BatchMetadata(
            scheduled=jax.device_put(scheduled, self._empty_sharding),
            query_start_loc=qsl,
            seq_lens=seq_lens,
            pages_tables=pt,
            padded_num_reqs=jax.device_put(numpy.int32(padded_num_reqs), self._empty_sharding),
            request_distribution=req_dist,
            logits_indices=logits_indices,
            input_ids_buf=input_ids_buf[:num_tokens_static],
            position_ids_buf=position_ids_buf[:num_tokens_static],
            num_requests=jax.device_put(numpy.int32(num_requests), self._empty_sharding),
            temperature=jax.device_put(temperature_cpu, self._empty_sharding),
            top_p=jax.device_put(top_p_cpu, self._empty_sharding),
            top_k=jax.device_put(top_k_cpu, self._empty_sharding),
            min_p=jax.device_put(min_p_cpu, self._empty_sharding),
            positions=jax.device_put(num_computed_tokens_cpu[:num_tokens_static], self._empty_sharding),
        )

        return metadata, input_ids_buf, position_ids_buf

    def get_model_step_fn(self) -> typing.Callable:
        """Create the model-only ejit that consumes precomputed metadata."""

        max_num_reqs = int(self.max_num_reqs)
        num_reqs_max_model_len = min(int(self.metadata.get_max_num_seqs()), max_num_reqs)

        metadata_sharding = BatchMetadata(
            scheduled=self._empty_sharding,
            query_start_loc=self._empty_sharding,
            seq_lens=self._empty_sharding,
            pages_tables=self._empty_sharding,
            padded_num_reqs=self._empty_sharding,
            request_distribution=self._empty_sharding,
            logits_indices=self._empty_sharding,
            input_ids_buf=self._empty_sharding,
            position_ids_buf=self._empty_sharding,
            num_requests=self._empty_sharding,
            temperature=self._empty_sharding,
            top_p=self._empty_sharding,
            top_k=self._empty_sharding,
            min_p=self._empty_sharding,
            positions=self._empty_sharding,
        )

        inputs_shardings = StepFunctionInputs(
            device_state=self._empty_sharding,
            kv_pages=es.extract_shardings(self.kv_pages, self.mesh),
            scheduled_full=self._empty_sharding,
            req_num_tokens_full=self._empty_sharding,
            active_mask_full=self._empty_sharding,
            rng_key=self._empty_sharding,
            batch_metadata=metadata_sharding,
        )

        outputs_shardings = ModelStepOutputs(
            kv_pages=es.extract_shardings(self.kv_pages, self.mesh),
            hidden_states=self._empty_sharding,
            logits=self._empty_sharding,
        )

        @ejit(
            static_argnums=(0,),
            donate_argnames=["inputs"],
            in_shardings=(
                es.extract_shardings(self.graphstate, self.mesh),
                es.extract_shardings(self.graphother, self.mesh),
                inputs_shardings,
            ),
            out_shardings=outputs_shardings,
        )
        def _model_step(
            graphdef,
            graphstate,
            graphother,
            inputs: StepFunctionInputs,
        ) -> ModelStepOutputs:
            metadata = inputs.batch_metadata
            kv_pages = inputs.kv_pages

            with self.model.mesh:
                model: EasyDeLBaseModule = nn.merge(graphdef, graphstate, graphother)
                input_ids_view = metadata.input_ids_buf
                position_ids_view = metadata.position_ids_buf

                cache_metadata = RaggedPagesMetadata(
                    pages_tables=metadata.pages_tables,
                    context_lens=metadata.seq_lens[:num_reqs_max_model_len],
                    query_start_loc=metadata.query_start_loc[: num_reqs_max_model_len + 1],
                    num_seqs=jnp.array([metadata.num_requests], dtype=jnp.int32),
                    num_slices_per_kv_cache_update_page=self.metadata.num_slices_per_kv_cache_update_page,
                    page_size=self.metadata.page_size,
                    request_distribution=metadata.request_distribution,
                )

                output = model(
                    input_ids=jnp.expand_dims(input_ids_view, 0),
                    position_ids=jnp.expand_dims(position_ids_view, 0),
                    past_key_values=kv_pages,
                    cache_metadata=cache_metadata,
                    apply_lm_head=False,
                )
                hs = output.last_hidden_state.squeeze(0)
                logits = model.apply_lm_head(hs[metadata.logits_indices])

                return ModelStepOutputs(
                    kv_pages=output.past_key_values,
                    hidden_states=hs,
                    logits=logits,
                )

        return _model_step

    def get_sampling_fn(self) -> typing.Callable:
        """Create the sampler/update ejit executed after model forward."""

        max_num_reqs = int(self.max_num_reqs)
        i_reqs = jnp.arange(max_num_reqs, dtype=jnp.int32)

        @ejit
        def _sampling_fn(
            metadata: BatchMetadata,
            device_state: DeviceSequenceState,
            req_num_tokens_full: jax.Array,
            active_mask_full: jax.Array,
            logits: jax.Array,
            rng_key: jax.Array,
        ):
            temp = metadata.temperature.reshape(-1, 1).astype(logits.dtype)
            topp = metadata.top_p.astype(logits.dtype)
            topk = metadata.top_k.astype(jnp.int32)
            minp = metadata.min_p.astype(logits.dtype)

            is_all_greedy = jnp.all(temp <= 0.0)
            need_min_p_sampling = jnp.any(minp > 0.0)

            sampling_metadata = SamplingMetadata(
                temperatures=temp,
                top_ps=topp,
                top_ks=topk,
                min_ps=minp,
                sampling_seeds=None,
                is_all_greedy=is_all_greedy,
                need_min_p_sampling=need_min_p_sampling,
                do_penalties=False,
                linear_penalty=None,
            )

            sampled_flat = sample_tokens(logits, sampling_metadata, metadata.positions, rng_key)
            total_tokens = metadata.query_start_loc[-1]
            rng_key = jax.random.fold_in(rng_key, jnp.int32(total_tokens))

            seq_lens_now_full = device_state.num_computed_tokens + metadata.scheduled
            meets_len_full = seq_lens_now_full >= req_num_tokens_full
            valid_mask_full = (
                (i_reqs < metadata.num_requests) & active_mask_full & (metadata.scheduled > 0) & meets_len_full
            )

            j_pos_full = jnp.clip(seq_lens_now_full, 0, self.max_model_len - 1)
            curr_vals_full = device_state.token_ids[i_reqs, j_pos_full]
            delta_full = jnp.where(valid_mask_full, sampled_flat - curr_vals_full, 0)

            token_ids = device_state.token_ids.at[(i_reqs, j_pos_full)].add(delta_full)
            num_tokens = device_state.num_tokens + valid_mask_full.astype(device_state.num_tokens.dtype)

            new_state = device_state.with_updates(token_ids=token_ids, num_tokens=num_tokens)
            out_tokens = jnp.where(valid_mask_full, sampled_flat, -1)
            return new_state, rng_key, out_tokens, valid_mask_full

        return _sampling_fn

    def _compile_model_step(self, num_tokens: int, compargs):
        """Compile the model-only ejit for specific token buckets."""
        mode = "aot" if self.use_aot_forward else "jit"
        key = (num_tokens, "model", mode)

        if key not in self._model_lowerd_history:
            if self.use_aot_forward:
                compiled = self._model_step_fn.lower(*compargs).compile()
                self._model_cache_put(key, compiled)
                warm_args = (compargs[1], compargs[2], compargs[3])
                self._debug_baselines[f"{num_tokens}_hash_in_model"] = _tree_hash(warm_args)
            else:

                def wrapped(graphstate, graphother, inputs):
                    return self._model_step_fn(self.graphdef, graphstate, graphother, inputs)

                _ = wrapped(self.graphstate, self.graphother, compargs[3])
                self._model_cache_put(key, wrapped)

    def _compile_sampler(self, num_tokens: int, inputs: StepFunctionInputs, metadata: BatchMetadata):
        """Compile the sampler/update ejit."""
        mode = "aot" if self.use_aot_forward else "jit"
        key = (num_tokens, "sampler", mode)

        if key in self._sampler_lowerd_history:
            return

        vocab_size = self.model.config.get_text_config().vocab_size
        dummy_logits = jax.device_put(
            jnp.zeros(
                (self.max_num_reqs, vocab_size),
                dtype=self.model.dtype,
            ),
            self._empty_sharding,
        )
        sampler_args = (
            metadata,
            inputs.device_state,
            inputs.req_num_tokens_full,
            inputs.active_mask_full,
            dummy_logits,
            inputs.rng_key,
        )

        if self.use_aot_forward:
            compiled = self._sampling_fn.lower(*sampler_args).compile()
            self._sampler_cache_put(key, compiled)
            self._debug_baselines[f"{num_tokens}_hash_in_sampler"] = _tree_hash(sampler_args)
        else:
            _ = self._sampling_fn(*sampler_args)
            self._sampler_cache_put(key, self._sampling_fn)

    def get_compiled_key(self, num_tokens: int, padded_num_reqs: int):
        """Retrieve pre-compiled model step function for given input dimensions.

        Args:
            num_tokens: Number of tokens in the input batch.
            padded_num_reqs: Padded number of requests for batching (unused in fused mode).

        Returns:
            Compiled fused step function for the specified number of tokens.
        """

        mode = "aot" if self.use_aot_forward else "jit"
        model_key = (num_tokens, "model", mode)
        sampler_key = (num_tokens, "sampler", mode)
        if model_key in self._model_lowerd_history:
            ...
        else:
            logger.warning(f"Cache miss for key={model_key}! Will trigger recompilation (model)")
            logger.warning(f"Available keys in cache: {list(self._model_lowerd_history.keys())}")
        if sampler_key in self._sampler_lowerd_history:
            ...
        else:
            logger.warning(f"Cache miss for key={sampler_key}! Will trigger recompilation (sampler)")
            logger.warning(f"Available keys in cache: {list(self._sampler_lowerd_history.keys())}")
        return self._model_cache_get(model_key), self._sampler_cache_get(sampler_key)

    def get_compile_configurations(
        self,
        kv_pages: RaggedPagesCache,
        rng_key: jax.random.PRNGKey,
        num_tokens: int,
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
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
            padded_num_reqs: Padded request count (unused).
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
        temp_buffer = SequenceBuffer.create(
            max_num_reqs=max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            vocab_size=self.model.config.get_text_config().vocab_size,
            page_sizes=[metadata.page_size],
            sharding=self._empty_sharding,
        )

        # Convert to device state
        device_state = temp_buffer.to_device_state(sharding=self._empty_sharding)

        # Create dummy inputs for prepare_batch_metadata (CPU arrays)
        scheduled_full_cpu = numpy.ones((max_num_reqs,), dtype=numpy.int32)
        active_mask_full_cpu = numpy.ones((max_num_reqs,), dtype=bool)
        input_ids_buf = jax.device_put(jnp.zeros((self.max_num_tokens,), dtype=jnp.int32), self._empty_sharding)
        position_ids_buf = jax.device_put(jnp.zeros((self.max_num_tokens,), dtype=jnp.int32), self._empty_sharding)

        # Get page table as CPU array
        page_table_cpu_dummy = numpy.asarray(jax.device_get(temp_buffer.page_table[0].get_array()))

        # Use prepare_batch_metadata to create metadata with correct shapes/shardings
        # This ensures compilation artifacts match runtime execution exactly
        dummy_metadata, input_ids_buf, position_ids_buf = self.prepare_batch_metadata(
            num_tokens_static=num_tokens,
            device_state=device_state,
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
        )

        # Convert to device for StepFunctionInputs
        scheduled_full = jax.device_put(jnp.asarray(scheduled_full_cpu), self._empty_sharding)
        active_mask_full = jax.device_put(jnp.asarray(active_mask_full_cpu), self._empty_sharding)

        inputs = StepFunctionInputs(
            device_state=device_state,
            kv_pages=kv_pages,
            scheduled_full=scheduled_full,
            req_num_tokens_full=jax.device_put(jnp.full((max_num_reqs,), 10, dtype=jnp.int32), self._empty_sharding),
            active_mask_full=active_mask_full,
            rng_key=jax.device_put(rng_key, self._empty_sharding),
            batch_metadata=dummy_metadata,
        )

        return [self.graphdef, self.graphstate, self.graphother, inputs]
