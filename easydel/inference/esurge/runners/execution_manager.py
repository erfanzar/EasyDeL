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
from .execution_types import StepFunctionInputs, StepFunctionOutputs
from .sequence_buffer import DeviceSequenceState, SequenceBuffer

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
        _map, tree, is_leaf=lambda x: isinstance(x, jax.Array | numpy.ndarray | int | float | bool)
    )


def _tree_hash_diff(tree_hash1, tree_hash2):
    def _map(p, t1, t2):
        p = key_path_to_str(p)
        oo = t1 == t2
        if not oo:
            print(f"p : {p} oo : {oo} t1 : {t1} t2 : {t2}")
        return oo

    return jax.tree_util.tree_map_with_path(_map, tree_hash1, tree_hash2, is_leaf=lambda x: isinstance(x, str))


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
        _step_fn: Base step function wrapper (ejit-decorated).
        _lowerd_history: OrderedDict LRU cache of compiled functions.
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

        self._empty_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())

        self.rng_key = jax.device_put(jax.random.PRNGKey(0), self._empty_sharding)

        self._step_fn: None | pjit.JitWrapped = None
        self._cache_capacity = 64
        self._lowerd_history = OrderedDict()
        self._debug_baselines = {}

        self.init_fns()

    def _cache_put(self, key, value):
        self._lowerd_history[key] = value
        self._lowerd_history.move_to_end(key)
        if len(self._lowerd_history) > self._cache_capacity:
            self._lowerd_history.popitem(last=False)

    def _cache_get(self, key):
        value = self._lowerd_history[key]
        self._lowerd_history.move_to_end(key)
        return value

    def clear_cache(self):
        self._lowerd_history.clear()

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
        scheduled_full: jax.Array,
        req_num_tokens_full: jax.Array,
        active_mask_full: jax.Array,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        padded_num_reqs: int,
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
        fn = self.get_compiled_key(num_tokens, padded_num_reqs)

        inputs = StepFunctionInputs(
            device_state=device_state,
            kv_pages=self.kv_pages,
            scheduled_full=scheduled_full,
            req_num_tokens_full=req_num_tokens_full,
            active_mask_full=active_mask_full,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            rng_key=self.rng_key,
        )

        result = fn(self.graphstate, self.graphother, inputs)

        device_state = result.device_state
        self.kv_pages = result.kv_pages
        input_ids_buf = result.input_ids_buf
        position_ids_buf = result.position_ids_buf
        query_start_loc_buf = result.query_start_loc
        seq_lens_buf = result.seq_lens
        pages_tables_buf = result.pages_tables
        self.rng_key = result.rng_key
        out_tokens_full = result.out_tokens
        valid_mask_full = result.valid_mask
        hidden_states = result.hidden_states
        logits = result.logits

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
        self.compile_key(num_tokens, padded_num_reqs, compargs)

    def init_fns(self) -> None:
        """Initialize the fused step execution function.

        Creates the fused execution function that combines input preparation,
        model forward pass, token sampling, and state updates. The function
        is wrapped with ejit for efficient execution.

        Note:
            Called automatically during initialization. Should not be called
            directly by users.
        """
        self._step_fn = self.get_step_fn()

    def get_step_fn(self) -> typing.Callable:
        """Create the fused step execution function.

        Builds a single ejit-wrapped function that performs a complete inference step:
        input preparation, model forward pass, token sampling, and state updates.
        This fusion minimizes host-device round-trips and enables XLA kernel fusion.

        The returned function signature is:
            fn(num_tokens: int, graphdef, graphstate, graphother, inputs: StepFunctionInputs)
            -> StepFunctionOutputs

        Where:
            - num_tokens: Static argument (compilation key) for token count bucket
            - graphdef: Static model structure
            - graphstate: Model weights (donated for in-place updates)
            - graphother: Auxiliary model state (donated)
            - inputs: StepFunctionInputs PyTree (donated)

        Returns:
            Callable fused step function with shardings applied for inputs/outputs.
            The function is decorated with ejit for efficient lowering and execution.

        Note:
            The function uses donate_argnames for memory efficiency. Input buffers
            are donated and may be mutated in-place by XLA.
        """
        max_num_reqs = int(self.max_num_reqs)
        num_reqs_max_model_len = min(int(self.metadata.get_max_num_seqs()), max_num_reqs)
        page_table_pad = jnp.int32(PAGE_TABLE_PADDING_VAL)

        i_reqs = jnp.arange(max_num_reqs, dtype=jnp.int32)
        i_rows_pt = jnp.arange(num_reqs_max_model_len, dtype=jnp.int32)

        inputs_shardings = StepFunctionInputs(
            device_state=self._empty_sharding,
            kv_pages=es.extract_shardings(self.kv_pages, self.mesh),
            scheduled_full=self._empty_sharding,
            req_num_tokens_full=self._empty_sharding,
            active_mask_full=self._empty_sharding,
            input_ids_buf=self._empty_sharding,
            position_ids_buf=self._empty_sharding,
            rng_key=self._empty_sharding,
        )

        outputs_shardings = StepFunctionOutputs(
            device_state=self._empty_sharding,
            kv_pages=es.extract_shardings(self.kv_pages, self.mesh),
            input_ids_buf=self._empty_sharding,
            position_ids_buf=self._empty_sharding,
            query_start_loc=self._empty_sharding,
            seq_lens=self._empty_sharding,
            pages_tables=self._empty_sharding,
            rng_key=self._empty_sharding,
            out_tokens=self._empty_sharding,
            valid_mask=self._empty_sharding,
            hidden_states=self._empty_sharding,
            logits=self._empty_sharding,
        )

        @ejit(
            static_argnums=(0, 1),
            donate_argnames=["inputs"],
            in_shardings=(
                es.extract_shardings(self.graphstate, self.mesh),  # graphstate
                es.extract_shardings(self.graphother, self.mesh),  # graphother
                inputs_shardings,  # StepFunctionInputs PyTree
            ),
            out_shardings=outputs_shardings,  # StepFunctionOutputs PyTree
        )
        def _fn(
            num_tokens_static: int,  # STATIC: padded_total bucket
            graphdef,
            graphstate,
            graphother,
            inputs: StepFunctionInputs,
        ) -> StepFunctionOutputs:
            device_state = inputs.device_state
            kv_pages = inputs.kv_pages
            scheduled_full = inputs.scheduled_full
            req_num_tokens_full = inputs.req_num_tokens_full
            active_mask_full = inputs.active_mask_full
            input_ids_buf = inputs.input_ids_buf
            position_ids_buf = inputs.position_ids_buf
            rng_key = inputs.rng_key

            with self.model.mesh:
                nr = jnp.minimum(jnp.int32(jnp.sum(active_mask_full)), jnp.int32(max_num_reqs))
                mask_reqs = i_reqs < nr
                scheduled = jnp.where(mask_reqs, scheduled_full, 0)

                cum = jnp.cumsum(scheduled)
                total = cum[-1]

                it = jnp.arange(num_tokens_static, dtype=jnp.int32)
                valid_tok = it < total
                req_for_tok = jnp.searchsorted(cum, it, side="right")
                req_for_tok = jnp.where(valid_tok, req_for_tok, 0)
                cum_prev = jnp.concatenate([jnp.zeros((1,), jnp.int32), cum[:-1]])
                base_pos = device_state.num_computed_tokens[req_for_tok]
                off_in_req = it - cum_prev[req_for_tok]
                positions = base_pos + off_in_req
                positions = jnp.where(valid_tok, positions, 0)

                in_ids = device_state.token_ids[req_for_tok, positions]
                in_ids = jnp.where(valid_tok, in_ids, 0)
                input_ids_buf = input_ids_buf.at[:num_tokens_static].set(in_ids)
                position_ids_buf = position_ids_buf.at[:num_tokens_static].set(positions)
                qsl = jnp.zeros((max_num_reqs + 1,), dtype=jnp.int32).at[1:].set(cum)
                seq_lens = jnp.where(mask_reqs, device_state.num_computed_tokens + scheduled, 0)

                pt_array = device_state.page_table[0].get_array()
                pt_src = pt_array[: min(pt_array.shape[0], num_reqs_max_model_len), :]
                mask_rows = i_rows_pt < jnp.minimum(nr, jnp.int32(num_reqs_max_model_len))
                pt = jnp.where(mask_rows[:, None], pt_src, page_table_pad)

                nr_safe = jnp.maximum(nr, 1)
                next_pow2 = jnp.left_shift(1, jnp.ceil(jnp.log2(nr_safe)).astype(jnp.int32))
                padded_num_reqs = jnp.where(
                    nr <= jnp.int32(self.min_input_pad),
                    jnp.int32(self.min_input_pad),
                    next_pow2,
                )
                padded_num_reqs = jnp.minimum(padded_num_reqs, jnp.int32(max_num_reqs))

                active_num_computed = jnp.where(mask_reqs, device_state.num_computed_tokens, 0)
                is_decode = (scheduled == 1) & (active_num_computed > 0)
                is_chunked_prefill = (scheduled > 1) & (active_num_computed > 0)
                decode_count = jnp.sum(is_decode.astype(jnp.int32))
                chunked_prefill_count = jnp.sum(is_chunked_prefill.astype(jnp.int32))
                boundary = jnp.minimum(decode_count + chunked_prefill_count, nr)
                request_distribution = jnp.array([decode_count, boundary, nr], dtype=jnp.int32)

                tmp_logits = cum - 1
                mask_logits = i_reqs < padded_num_reqs
                logits_indices = jnp.where(mask_logits, tmp_logits, 0)

                input_ids_view = input_ids_buf[:num_tokens_static]
                position_ids_view = position_ids_buf[:num_tokens_static]

                model: EasyDeLBaseModule = nn.merge(graphdef, graphstate, graphother)
                output = model(
                    input_ids=jnp.expand_dims(input_ids_view, 0),
                    position_ids=jnp.expand_dims(position_ids_view, 0),
                    past_key_values=kv_pages,
                    cache_metadata=RaggedPagesMetadata(
                        pages_tables=pt,
                        context_lens=seq_lens[:num_reqs_max_model_len],
                        query_start_loc=qsl[: num_reqs_max_model_len + 1],
                        num_seqs=jnp.array([nr], dtype=jnp.int32),
                        num_slices_per_kv_cache_update_page=self.metadata.num_slices_per_kv_cache_update_page,
                        page_size=self.metadata.page_size,
                        request_distribution=request_distribution,
                    ),
                    apply_lm_head=False,
                )
                hs = output.last_hidden_state.squeeze(0)
                logits = model.apply_lm_head(hs[logits_indices])

                temp = device_state.temperature[:max_num_reqs].reshape(-1, 1).astype(logits.dtype)
                topp = device_state.top_p[:max_num_reqs].astype(logits.dtype)
                topk = device_state.top_k[:max_num_reqs].astype(jnp.int32)
                minp = device_state.min_p[:max_num_reqs].astype(logits.dtype)

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

                positions = device_state.num_computed_tokens[:max_num_reqs]
                sampled_flat = sample_tokens(logits, sampling_metadata, positions, rng_key)
                rng_key = jax.random.fold_in(rng_key, jnp.int32(num_tokens_static))

                seq_lens_now_full = device_state.num_computed_tokens + scheduled
                meets_len_full = seq_lens_now_full >= req_num_tokens_full
                valid_mask_full = (i_reqs < nr) & active_mask_full & (scheduled > 0) & meets_len_full

                j_pos_full = jnp.clip(seq_lens_now_full, 0, self.max_model_len - 1)
                curr_vals_full = device_state.token_ids[i_reqs, j_pos_full]
                delta_full = jnp.where(valid_mask_full, sampled_flat - curr_vals_full, 0)

                token_ids = device_state.token_ids.at[(i_reqs, j_pos_full)].add(delta_full)
                num_tokens = device_state.num_tokens + valid_mask_full.astype(device_state.num_tokens.dtype)

                return StepFunctionOutputs(
                    device_state=device_state.with_updates(token_ids=token_ids, num_tokens=num_tokens),
                    kv_pages=output.past_key_values,
                    input_ids_buf=input_ids_buf,
                    position_ids_buf=position_ids_buf,
                    query_start_loc=qsl,
                    seq_lens=seq_lens,
                    pages_tables=pt,
                    rng_key=rng_key,
                    out_tokens=jnp.where(valid_mask_full, sampled_flat, -1),
                    valid_mask=valid_mask_full,
                    hidden_states=hs,
                    logits=logits,
                )

        return _fn

    def compile_key(self, num_tokens: int, padded_num_reqs: int, compargs):
        """Compile fused step execution function for specific input dimensions.

        Handles both AOT and JIT compilation modes based on use_aot_forward flag.
        For AOT mode (default), pre-compiles functions using JAX's lower/compile API.
        For JIT mode, executes the function once to trigger JIT compilation and caches
        the wrapped function.

        Args:
            num_tokens: Number of tokens in the input batch.
            padded_num_reqs: Padded number of requests for batching (unused in fused mode).
            compargs: Compilation arguments tuple where compargs contains fused step args.
        """
        mode = "aot" if self.use_aot_forward else "jit"
        fused_key = (num_tokens, "fused", mode)

        if fused_key not in self._lowerd_history:
            if self.use_aot_forward:
                compiled = self._step_fn.lower(num_tokens, *compargs).compile()
                self._cache_put(fused_key, compiled)
                warm_args = (compargs[1], compargs[2], compargs[3])
                self._debug_baselines[f"{num_tokens}_hash_in"] = _tree_hash(warm_args)
            else:
                partial_fn = partial(self._step_fn, num_tokens, self.graphdef)
                result = partial_fn(self.graphstate, self.graphother, compargs[3])
                _device_state = result.device_state
                self.kv_pages = result.kv_pages
                _input_ids_buf = result.input_ids_buf
                _position_ids_buf = result.position_ids_buf
                _query_start_loc_buf = result.query_start_loc
                _seq_lens_buf = result.seq_lens
                _pages_tables_buf = result.pages_tables
                _rng_key = result.rng_key
                _out_tokens_full = result.out_tokens
                _valid_mask_full = result.valid_mask
                _hidden_states = result.hidden_states
                _logits = result.logits

                self._cache_put(fused_key, partial_fn)

    def get_compiled_key(self, num_tokens: int, padded_num_reqs: int):
        """Retrieve pre-compiled fused step function for given input dimensions.

        Args:
            num_tokens: Number of tokens in the input batch.
            padded_num_reqs: Padded number of requests for batching (unused in fused mode).

        Returns:
            Compiled fused step function for the specified number of tokens.
        """

        mode = "aot" if self.use_aot_forward else "jit"
        key = (num_tokens, "fused", mode)

        if key in self._lowerd_history:
            ...
        else:
            logger.warning(f"Cache miss for key={key}! Will trigger recompilation")
            logger.warning(f"Available keys in cache: {list(self._lowerd_history.keys())}")
        return self._cache_get(key)

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

        temp_buffer = SequenceBuffer.create(
            max_num_reqs=max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            vocab_size=self.model.config.get_text_config().vocab_size,
            page_sizes=[metadata.page_size],
        )
        device_state = temp_buffer.to_device_state()

        inputs = StepFunctionInputs(
            device_state=_device_put_tree_uniform(device_state, self._empty_sharding),
            kv_pages=kv_pages,
            scheduled_full=jax.device_put(jnp.ones((max_num_reqs,), dtype=jnp.int32), self._empty_sharding),
            req_num_tokens_full=jax.device_put(jnp.full((max_num_reqs,), 10, dtype=jnp.int32), self._empty_sharding),
            active_mask_full=jax.device_put(jnp.ones((max_num_reqs,), dtype=bool), self._empty_sharding),
            input_ids_buf=jax.device_put(jnp.zeros((self.max_num_tokens,), dtype=jnp.int32), self._empty_sharding),
            position_ids_buf=jax.device_put(jnp.zeros((self.max_num_tokens,), dtype=jnp.int32), self._empty_sharding),
            rng_key=jax.device_put(rng_key, self._empty_sharding),
        )

        return [self.graphdef, self.graphstate, self.graphother, inputs]
