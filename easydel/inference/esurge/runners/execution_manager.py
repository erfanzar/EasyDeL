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

"""Execution manager for efficient model inference with precompiled functions.

This module provides the ExecutionManager class that handles compilation and caching
of model execution functions for different batch sizes and token counts. It supports
both AOT (Ahead-of-Time) and JIT (Just-In-Time) compilation strategies for optimal
performance in production and development environments.

The manager supports three execution modes:
    - Combined forward: Single function for both hidden states and token generation
    - Separate functions: Separate functions for hidden states and token generation
    - Fused step: Single function combining prepare_inputs, forward, sampling, and apply_token

Example:
    >>> from easydel.inference.esurge.runners import ExecutionManager
    >>> executor = ExecutionManager(
    ...     model=my_model,
    ...     mesh=device_mesh,
    ...     kv_pages=cache_pages,
    ...     use_combined_forward=True,
    ...     use_aot_forward=True
    ... )
    >>> executor.compile(token_paddings, ...)
    >>> tokens = executor.execute(inputs, ...)
"""

from __future__ import annotations

import time
import typing
from functools import partial

import jax
from eformer import escale as es
from eformer.loggings import ProgressLogger, get_logger
from flax import nnx as nn
from jax import numpy as jnp
from jax._src import pjit

from easydel.layers.caching import PagesCache, PagesCacheMetaData, PagesMetadata
from easydel.utils import ejit

from ...vsurge.core.functions import sample_top_p_efficient
from ..page_table import PAGE_TABLE_PADDING_VAL, SLOT_MAPPING_PADDING_VAL
from .sequence_buffer import DeviceSequenceState, ModelRunnerSamplingMetadata

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


class ExecutionManager:
    """Manages precompiled execution functions for efficient model inference.

    This class handles the compilation and caching of model execution functions
    for different batch sizes and token counts. It supports two execution modes:
    - Combined forward: Single function for both hidden states and token generation
    - Separate functions: Separate functions for hidden states and token generation

    The manager supports both AOT (Ahead-of-Time) and JIT (Just-In-Time) compilation:
    - AOT mode (default): Pre-compiles functions using JAX's lower/compile API for
      optimal performance in production
    - JIT mode: Compiles functions on first use with graph definition as static
      argument, more flexible for development

    The manager pre-compiles functions for various configurations to avoid
    runtime compilation overhead, enabling seamless switching between different
    batch sizes and sequence lengths.

    Attributes:
        model: The EasyDeL model being managed.
        mesh: JAX sharding mesh for distributed execution.
        kv_pages: KV cache pages for attention.
        use_combined_forward: Whether to use combined or separate functions.
        use_aot_forward: Whether to use AOT compilation (default: True).
        graphdef, graphstate, graphother: Split model components for JAX.
        _lowerd_history: Cache of compiled functions.

    Example:
        >>> executor = ExecutionManager(
        ...     model=my_model,
        ...     mesh=device_mesh,
        ...     kv_pages=cache_pages,
        ...     use_combined_forward=True,
        ...     use_aot_forward=True  # Use AOT compilation
        ... )
        >>> executor.compile(token_paddings, ...)
        >>> tokens = executor.execute(inputs, ...)
    """

    def __init__(
        self,
        model: EasyDeLBaseModule,
        mesh: jax.sharding.Mesh,
        kv_pages: PagesCache,
        use_combined_forward: bool = False,
        use_aot_forward: bool = True,
        use_fused_step: bool = False,
        min_input_pad: int = 8,
        max_model_len: int = 2**13,
        max_num_reqs: int = 16,
        max_num_tokens: int | None = None,
        metadata: PagesCacheMetaData = None,
    ):
        """Initialize the executor manager.

        Args:
            model: The EasyDeL model instance.
            mesh: JAX sharding mesh for distributed execution.
            kv_pages: Pages cache for KV cache management.
            use_combined_forward: Whether to use combined forward pass for model and token
                generation in a single function call. Default is False.
            use_aot_forward: Whether to use Ahead-of-Time (AOT) compilation for model
                execution. When True (default), functions are pre-compiled for better
                performance. When False, uses Just-In-Time (JIT) compilation with
                the graph definition passed as a static argument.
            use_fused_step: Whether to use fused step that combines prepare_inputs, forward,
                sampling, and apply_token in a single function. Default is False.
            min_input_pad: Minimum padding for inputs.
            max_model_len: Maximum model sequence length.
            max_num_reqs: Maximum number of requests.
            max_num_tokens: Maximum number of tokens for batching.
            metadata: Pages cache metadata.
        """
        logger.info(f"Initializing ExecutionManager with {use_combined_forward=}, {use_fused_step=}")
        self.model = model
        self.mesh = mesh
        self.kv_pages = kv_pages
        self.use_combined_forward = use_combined_forward
        self.use_aot_forward = use_aot_forward
        self.use_fused_step = use_fused_step
        self.min_input_pad = min_input_pad
        self.max_model_len = max_model_len
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens if max_num_tokens is not None else max_model_len
        self.metadata = metadata
        logger.debug("Splitting model module for graph-based execution")
        self.graphdef, self.graphstate, self.graphother = model.split_module()

        self.rng_key = jax.random.PRNGKey(0)

        self._empty_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())

        self._main_fn: None | pjit.JitWrapped = None
        self._compute_hidden_states_fn: None | pjit.JitWrapped = None
        self._compute_tokens_fn: None | pjit.JitWrapped = None
        self._fused_step_fn: None | pjit.JitWrapped = None

        self._lowerd_history = dict()

        logger.debug("Initializing execution functions")
        self.init_fns()
        logger.debug("ExecutionManager initialization complete")

    def execute_fused(
        self,
        num_tokens: int,
        dev_state: DeviceSequenceState,
        scheduled_full: jax.Array,
        req_num_tokens_full: jax.Array,
        active_mask_full: jax.Array,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        slot_mapping_buf: jax.Array,
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
    ]:
        """Execute the fused step function.

        This method runs a single fused execution step that combines input preparation,
        model forward pass, token sampling, and state updates into a single compiled
        function for maximum efficiency.

        Args:
            num_tokens: Number of tokens to process in this batch.
            dev_state: Current device sequence state containing token IDs and metadata.
            scheduled_full: Array of scheduled tokens per request [max_num_reqs].
            req_num_tokens_full: Array of required number of tokens per request [max_num_reqs].
            active_mask_full: Boolean mask indicating active requests [max_num_reqs].
            input_ids_buf: Buffer for input token IDs [max_num_tokens].
            position_ids_buf: Buffer for position IDs [max_num_tokens].
            query_start_loc_buf: Buffer for query start locations [max_num_reqs+1].
            seq_lens_buf: Buffer for sequence lengths [max_num_reqs].
            pages_tables_buf: Buffer for page tables [num_reqs_max_model_len, max_pages_per_req].
            slot_mapping_buf: Buffer for slot mapping [3, max_padded_slices].
            padded_num_reqs: Padded number of requests for compilation efficiency.

        Returns:
            A tuple containing:
                - dev_state: Updated device sequence state.
                - out_tokens_full: Generated token IDs for each request.
                - valid_mask_full: Mask indicating which tokens are valid.
                - input_ids_buf: Updated input IDs buffer.
                - position_ids_buf: Updated position IDs buffer.
                - query_start_loc_buf: Updated query start locations buffer.
                - seq_lens_buf: Updated sequence lengths buffer.
                - pages_tables_buf: Updated page tables buffer.
                - slot_mapping_buf: Updated slot mapping buffer.

        Raises:
            KeyError: If no compiled function exists for the given configuration.

        Note:
            This method requires use_fused_step=True during initialization.
        """
        fn = self.get_compiled_key(num_tokens, padded_num_reqs)
        if self.use_aot_forward:
            # AOT: function is already compiled, no static arguments needed
            result = fn(
                self.graphstate,
                self.graphother,
                dev_state,
                self.kv_pages,
                scheduled_full,
                req_num_tokens_full,
                active_mask_full,
                input_ids_buf,
                position_ids_buf,
                slot_mapping_buf,
                self.rng_key,
            )
        else:
            # JIT: pass num_tokens and graphdef as static arguments
            result = fn(
                num_tokens,
                self.graphdef,
                self.graphstate,
                self.graphother,
                dev_state,
                self.kv_pages,
                scheduled_full,
                req_num_tokens_full,
                active_mask_full,
                input_ids_buf,
                position_ids_buf,
                slot_mapping_buf,
                self.rng_key,
            )

        # Update internal state and return all buffers
        (
            dev_state,
            self.kv_pages,
            input_ids_buf,
            position_ids_buf,
            query_start_loc_buf,
            seq_lens_buf,
            pages_tables_buf,
            slot_mapping_buf,
            self.rng_key,
            out_tokens_full,
            valid_mask_full,
        ) = result

        return (
            dev_state,
            out_tokens_full,
            valid_mask_full,
            input_ids_buf,
            position_ids_buf,
            query_start_loc_buf,
            seq_lens_buf,
            pages_tables_buf,
            slot_mapping_buf,
        )

    def execute(
        self,
        input_ids_view: jax.Array,
        position_ids_view: jax.Array,
        cache_metadata: PagesMetadata,
        logits_indices: jax.Array,
        sampling_metadata: ModelRunnerSamplingMetadata,
        padded_num_reqs: int,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Execute the model on prepared inputs.

        Selects and runs the appropriate pre-compiled function based on
        input shapes. Handles both combined and separate execution modes.

        When AOT compilation is disabled (use_aot_forward=False), the graph
        definition is passed as a static argument during execution for JIT
        compilation. When enabled (default), pre-compiled functions are used
        for better performance.

        Args:
            input_ids_view: Token IDs to process [num_tokens].
            position_ids_view: Position IDs for tokens [num_tokens].
            cache_metadata: Paged attention metadata.
            logits_indices: Indices for logit extraction.
            sampling_metadata: Parameters for token sampling.
            padded_num_reqs: Padded number of requests.

        Returns:
            tuple: (sampled_token_ids, logits or None)
                - sampled_token_ids: Generated token IDs.
                - logits: Raw logits (only in separate mode).
        """
        if self.use_fused_step:
            raise ValueError("Use execute_fused for fused step execution")
        static_arguments = (self.graphdef,) if not self.use_aot_forward else ()
        if self.use_combined_forward:
            fn = self.get_compiled_key(input_ids_view.shape[0], padded_num_reqs)
            token_ids, self.kv_pages, self.rng_key = fn(
                *static_arguments,
                self.graphstate,
                self.graphother,
                input_ids_view,
                position_ids_view,
                self.kv_pages,
                cache_metadata,
                logits_indices,
                sampling_metadata,
                self.rng_key,
            )
            return token_ids, None
        else:
            hfn, tfn = self.get_compiled_key(input_ids_view.shape[0], padded_num_reqs)
            hidden_states, self.kv_pages = hfn(
                *static_arguments,
                self.graphstate,
                self.graphother,
                input_ids_view,
                position_ids_view,
                self.kv_pages,
                cache_metadata,
            )
            token_ids, self.rng_key = tfn(
                *static_arguments,
                self.graphstate,
                self.graphother,
                hidden_states,
                logits_indices,
                sampling_metadata,
                self.rng_key,
            )
            return token_ids, token_ids

    def compile(
        self,
        num_tokens_paddings: list[int],
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
        max_num_reqs: int,
        metadata: PagesCacheMetaData,
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
        logger.debug(f"Starting compilation for {len(num_tokens_paddings)} token padding sizes")
        logger.debug(f"Token paddings: {num_tokens_paddings}")
        logger.debug(f"Max pages per request: {max_pages_per_req}, Max requests: {max_num_reqs}")

        ufn = partial(_get_padded_num_reqs_with_upper_limit, min_input_pad=self.min_input_pad)
        reqs_padds = list(set([ufn(num_reqs, max_num_reqs) for num_reqs in range(max_num_reqs)]))
        total_compilations = len(num_tokens_paddings) * len(reqs_padds)
        compilation_count = 0

        # Use the new ProgressLogger
        progress = ProgressLogger("eSurge", logger)

        for num_tokens in num_tokens_paddings:
            for reqs_padd in reqs_padds:
                compile_start = time.time()

                # Update progress
                progress_msg = (
                    f"Compiling [{compilation_count + 1}/{total_compilations}]:"
                    f" {num_tokens:5d} tokens, {reqs_padd:2d} padded requests"
                )
                progress.update(compilation_count, total_compilations, progress_msg)

                self._step_compile(
                    num_tokens=num_tokens,
                    num_reqs_max_model_len=num_reqs_max_model_len,
                    max_pages_per_req=max_pages_per_req,
                    max_num_reqs=max_num_reqs,
                    padded_num_reqs=reqs_padd,
                    metadata=metadata,
                )
                compile_time = time.time() - compile_start
                logger.debug(f"Step completed in {compile_time:.2f}s")
                compilation_count += 1

        # Complete the progress
        progress.complete(f"All {total_compilations} compilations completed")

    def _step_compile(
        self,
        num_tokens: int,
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
        max_num_reqs: int,
        padded_num_reqs: int,
        metadata: PagesCacheMetaData,
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
        """Initialize all execution functions based on configuration.

        Creates the appropriate execution functions based on the execution mode
        (combined, separate, or fused). Functions are wrapped with ejit for
        efficient execution.

        Note:
            Called automatically during initialization. Should not be called
            directly by users.
        """
        self._main_fn = self.get_fn()
        self._compute_hidden_states_fn = self.get_compute_hidden_states_fn()
        self._compute_tokens_fn = self.get_compute_tokens_fn()
        self._fused_step_fn = self.get_fused_step_fn()

    def get_compute_hidden_states_fn(self) -> typing.Callable:
        """Create function for computing model hidden states.

        Returns:
            A callable that computes hidden states from input tokens without
            applying the language model head. The function is wrapped with ejit
            for efficient execution.

        Note:
            This function is used in separate execution mode where hidden states
            and token generation are computed in two steps.
        """

        @ejit(
            static_argnums=(0,),
            donate_argnames=["input_ids", "position_ids", "kv_pages"],
            in_shardings=(
                es.extract_shardings(self.graphstate, self.mesh),
                es.extract_shardings(self.graphother, self.mesh),
                self._empty_sharding,  # input_ids
                self._empty_sharding,  # position_ids
                es.extract_shardings(self.kv_pages, self.mesh),  # kv_pages
                self._empty_sharding,  # cache_metadata
            ),
            out_shardings=(self._empty_sharding, es.extract_shardings(self.kv_pages, self.mesh)),
        )
        def _fn(
            graphdef,
            graphstate,
            graphother,
            input_ids: jax.Array,
            position_ids: jax.Array,
            kv_pages: PagesCache,
            cache_metadata: PagesMetadata,
        ):
            model: EasyDeLBaseModule = nn.merge(graphdef, graphstate, graphother)
            with model.mesh:
                output = model(
                    input_ids=jnp.expand_dims(input_ids, 0),
                    position_ids=jnp.expand_dims(position_ids, 0),
                    past_key_values=kv_pages,
                    cache_metadata=cache_metadata,
                    apply_lm_head=False,
                )
                return output.last_hidden_state.squeeze(0), output.past_key_values

        return _fn

    def get_compute_tokens_fn(self) -> typing.Callable:
        """Create function for generating tokens from hidden states.

        Returns:
            A callable that applies the language model head to hidden states
            and performs token sampling. The function is wrapped with ejit
            for efficient execution.

        Note:
            This function is used in separate execution mode where hidden states
            and token generation are computed in two steps.
        """

        @ejit(
            static_argnums=(0,),
            in_shardings=(
                es.extract_shardings(self.graphstate, self.mesh),
                es.extract_shardings(self.graphother, self.mesh),
                self._empty_sharding,  # hidden_states
                self._empty_sharding,  # logits_indices
                self._empty_sharding,  # sampling_params
                self._empty_sharding,  # rng_key
            ),
            out_shardings=(self._empty_sharding, self._empty_sharding),
        )
        def _fn(
            graphdef,
            graphstate,
            graphother,
            hidden_states: jax.Array,
            logits_indices: jax.Array,
            sampling_params: ModelRunnerSamplingMetadata,
            rng_key: jax.random.PRNGKey,
        ):
            model: EasyDeLBaseModule = nn.merge(graphdef, graphstate, graphother)
            with model.mesh:
                logits = model.apply_lm_head(hidden_states[logits_indices])
                keys = jax.random.split(rng_key, logits.shape[0] + 1)
                samples = jax.vmap(sample_top_p_efficient, in_axes=(0, 0, 0, 0, None), out_axes=0)(
                    logits,
                    sampling_params.top_p.astype(logits.dtype),
                    sampling_params.temperature.astype(logits.dtype),
                    keys[1:],
                    64,
                )
                return samples.reshape(-1, 1), keys[0]

        return _fn

    def get_fused_step_fn(self) -> typing.Callable:
        """Create the fused step function.

        Creates a single function that combines input preparation, model forward pass,
        token sampling, and state updates. This provides the best performance by
        minimizing host-device communication and maximizing kernel fusion.

        Returns:
            A callable that performs a complete inference step. The function is
            wrapped with ejit for efficient execution.

        Note:
            This function is only created when use_fused_step=True. It provides
            the most efficient execution path for production inference.
        """
        max_num_reqs = int(self.max_num_reqs)
        page_size = int(self.metadata.page_size)
        max_pages_per_req = int(self.metadata.max_num_pages_per_req)
        num_reqs_max_model_len = min(int(self.metadata.get_max_num_seqs()), max_num_reqs)
        slices_per_page = int(self.metadata.num_slices_per_kv_cache_update_page)
        page_table_pad = jnp.int32(PAGE_TABLE_PADDING_VAL)
        slot_mapping_pad = jnp.int32(SLOT_MAPPING_PADDING_VAL)
        max_num_tokens = int(self.max_model_len)
        max_padded_slices = int(self.metadata.get_padded_num_slices(max_num_tokens, max_num_reqs))

        i_reqs = jnp.arange(max_num_reqs, dtype=jnp.int32)
        i_rows_pt = jnp.arange(num_reqs_max_model_len, dtype=jnp.int32)
        i_slices = jnp.arange(max_padded_slices, dtype=jnp.int32)

        @ejit(
            static_argnums=(0, 1),
            donate_argnames=[
                "dev_state",
                "kv_pages",
                "input_ids_buf",
                "position_ids_buf",
                "slot_mapping_buf",
            ],
            in_shardings=(
                es.extract_shardings(self.graphstate, self.mesh),  # graphstate
                es.extract_shardings(self.graphother, self.mesh),  # graphother
                self._empty_sharding,  # dev_state (PyTree)
                es.extract_shardings(self.kv_pages, self.mesh),  # kv_pages
                self._empty_sharding,  # scheduled_full
                self._empty_sharding,  # req_num_tokens_full
                self._empty_sharding,  # active_mask_full
                self._empty_sharding,  # input_ids_buf
                self._empty_sharding,  # position_ids_buf
                self._empty_sharding,  # slot_mapping_buf
                self._empty_sharding,  # rng_key
            ),
            out_shardings=(
                self._empty_sharding,  # dev_state (updated)
                es.extract_shardings(self.kv_pages, self.mesh),  # kv_pages
                self._empty_sharding,  # input_ids_buf
                self._empty_sharding,  # position_ids_buf
                self._empty_sharding,  # slot_mapping_buf
                self._empty_sharding,  # query_start_loc_buf
                self._empty_sharding,  # seq_lens_buf
                self._empty_sharding,  # pages_tables_buf
                self._empty_sharding,  # rng_key
                self._empty_sharding,  # out_tokens (full-size, masked)
                self._empty_sharding,  # valid_mask (full-size)
            ),
        )
        def _fn(
            num_tokens_static: int,  # STATIC: padded_total bucket
            graphdef,
            graphstate,
            graphother,
            dev_state: DeviceSequenceState,
            kv_pages: PagesCache,
            scheduled_full: jax.Array,  # [max_num_reqs] int32
            req_num_tokens_full: jax.Array,  # [max_num_reqs] int32
            active_mask_full: jax.Array,  # [max_num_reqs] bool
            input_ids_buf: jax.Array,  # [max_num_tokens]
            position_ids_buf: jax.Array,  # [max_num_tokens]
            slot_mapping_buf: jax.Array,  # [3, max_padded_slices]
            rng_key: jax.Array,
        ):
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
            base_pos = dev_state.num_computed_tokens[req_for_tok]
            off_in_req = it - cum_prev[req_for_tok]
            positions = base_pos + off_in_req
            positions = jnp.where(valid_tok, positions, 0)

            in_ids = dev_state.token_ids[req_for_tok, positions]
            in_ids = jnp.where(valid_tok, in_ids, 0)
            input_ids_buf = input_ids_buf.at[:num_tokens_static].set(in_ids)
            position_ids_buf = position_ids_buf.at[:num_tokens_static].set(positions)
            qsl = jnp.zeros((max_num_reqs + 1,), dtype=jnp.int32).at[1:].set(cum)
            seq_lens = jnp.where(mask_reqs, dev_state.num_computed_tokens + scheduled, 0)
            pt_array = dev_state.page_table[0].get_array()
            pt_src = pt_array[: min(pt_array.shape[0], num_reqs_max_model_len), :]
            mask_rows = i_rows_pt < jnp.minimum(nr, jnp.int32(num_reqs_max_model_len))
            pt = jnp.where(mask_rows[:, None], pt_src, page_table_pad)
            s = dev_state.num_computed_tokens
            e = s + scheduled
            lps = s // page_size
            lpe = (jnp.maximum(e, 1) - 1) // page_size
            page_lens = jnp.where(scheduled > 0, lpe - lps + 1, 0)
            page_cum = jnp.cumsum(page_lens)
            total_pages = page_cum[-1]
            sp = jnp.int32(slices_per_page)
            padded_num_slices = jnp.minimum(((total_pages + sp - 1) // sp) * sp, jnp.int32(max_padded_slices))

            valid_slice = i_slices < total_pages
            within_pad = i_slices < padded_num_slices
            slice_active = valid_slice & within_pad

            page_cum_prev = jnp.concatenate([jnp.zeros((1,), jnp.int32), page_cum[:-1]])
            req_for_slice = jnp.searchsorted(page_cum, i_slices, side="right")
            req_for_slice = jnp.where(slice_active, req_for_slice, 0)
            local_off = i_slices - page_cum_prev[req_for_slice]
            pt_full = pt.reshape((-1,))
            gpi = req_for_slice * jnp.int32(max_pages_per_req) + lps[req_for_slice] + local_off
            gpi_safe = jnp.clip(gpi, 0, jnp.int32(pt_full.size - 1))
            page_numbers = jnp.where(slice_active, pt_full[gpi_safe], 0)

            s_mod = s % page_size
            e_mod = ((jnp.maximum(e, 1) - 1) % page_size) + 1
            lens_rep = page_lens[req_for_slice]

            is_first = local_off == 0
            is_last = local_off == (lens_rep - 1)

            kv_local_st = jnp.where(is_first, s_mod[req_for_slice], 0)
            kv_local_en = jnp.where(is_last, e_mod[req_for_slice], jnp.int32(page_size))
            slice_lens = jnp.maximum(kv_local_en - kv_local_st, 0)
            kv_cache_start = kv_local_st + page_numbers * page_size

            slice_lens_masked = jnp.where(slice_active, slice_lens, 0)
            csl = jnp.cumsum(slice_lens_masked)
            new_kv_start = jnp.where(slice_active, jnp.roll(csl, 1).at[0].set(0), 0)
            slot_mapping_buf = slot_mapping_buf.at[0, :].set(jnp.where(slice_active, kv_cache_start, slot_mapping_pad))
            slot_mapping_buf = slot_mapping_buf.at[1, :].set(jnp.where(slice_active, new_kv_start, slot_mapping_pad))
            slot_mapping_buf = slot_mapping_buf.at[2, :].set(jnp.where(slice_active, slice_lens, slot_mapping_pad))
            nr_safe = jnp.maximum(nr, 1)
            next_pow2 = jnp.left_shift(1, jnp.ceil(jnp.log2(nr_safe)).astype(jnp.int32))
            padded_num_reqs = jnp.where(nr <= jnp.int32(self.min_input_pad), jnp.int32(self.min_input_pad), next_pow2)
            padded_num_reqs = jnp.minimum(padded_num_reqs, jnp.int32(max_num_reqs))

            # logits indices from cum
            tmp_logits = cum - 1
            mask_logits = i_reqs < padded_num_reqs
            logits_indices = jnp.where(mask_logits, tmp_logits, 0)

            # 2) Forward + sampling (unchanged below)
            input_ids_view = input_ids_buf[:num_tokens_static]
            position_ids_view = position_ids_buf[:num_tokens_static]

            model: EasyDeLBaseModule = nn.merge(graphdef, graphstate, graphother)
            with model.mesh:
                output = model(
                    input_ids=jnp.expand_dims(input_ids_view, 0),
                    position_ids=jnp.expand_dims(position_ids_view, 0),
                    past_key_values=kv_pages,
                    cache_metadata=PagesMetadata(
                        pages_tables=pt,
                        slot_mapping=slot_mapping_buf,
                        context_lens=seq_lens[:num_reqs_max_model_len],
                        query_start_loc=qsl[: num_reqs_max_model_len + 1],
                        num_seqs=jnp.array([nr], dtype=jnp.int32),
                        num_kv_update_slices=jnp.array([total_pages], dtype=jnp.int32),
                        num_slices_per_kv_cache_update_page=self.metadata.num_slices_per_kv_cache_update_page,
                        page_size=self.metadata.page_size,
                    ),
                    apply_lm_head=False,
                )
                hs = output.last_hidden_state.squeeze(0)
                logits = model.apply_lm_head(hs[logits_indices])

            temp = dev_state.temperature[:max_num_reqs].astype(logits.dtype)
            topp = dev_state.top_p[:max_num_reqs].astype(logits.dtype)
            keys = jax.random.split(rng_key, logits.shape[0] + 1)
            samples = jax.vmap(sample_top_p_efficient, in_axes=(0, 0, 0, 0, None), out_axes=0)(
                logits, topp, temp, keys[1:], 64
            )
            sampled_flat = samples.reshape(-1)

            # 3) Apply tokens
            seq_lens_now_full = dev_state.num_computed_tokens + scheduled
            meets_len_full = seq_lens_now_full >= req_num_tokens_full
            valid_mask_full = (i_reqs < nr) & active_mask_full & (scheduled > 0) & meets_len_full

            j_pos_full = jnp.clip(seq_lens_now_full, 0, self.max_model_len - 1)
            curr_vals_full = dev_state.token_ids[i_reqs, j_pos_full]
            delta_full = jnp.where(valid_mask_full, sampled_flat - curr_vals_full, 0)

            token_ids = dev_state.token_ids.at[(i_reqs, j_pos_full)].add(delta_full)
            num_tokens = dev_state.num_tokens + valid_mask_full.astype(dev_state.num_tokens.dtype)

            dev_state = dev_state.with_updates(token_ids=token_ids, num_tokens=num_tokens)

            out_tokens_full = jnp.where(valid_mask_full, sampled_flat, -1)
            return (
                dev_state,
                output.past_key_values,
                input_ids_buf,
                position_ids_buf,
                qsl,
                seq_lens,
                pt,
                slot_mapping_buf,
                keys[0],
                out_tokens_full,
                valid_mask_full,
            )

        return _fn

    def get_fn(self) -> typing.Callable:
        """Create the combined forward pass and token generation function.

        Returns:
            A callable that performs both forward pass and token generation
            in a single function call. The function is wrapped with ejit
            for efficient execution.

        Note:
            This function is used when use_combined_forward=True and
            use_fused_step=False.
        """

        @ejit(
            static_argnums=(0,),
            donate_argnames=["input_ids", "position_ids", "kv_pages"],
            in_shardings=(
                es.extract_shardings(self.graphstate, self.mesh),
                es.extract_shardings(self.graphother, self.mesh),
                self._empty_sharding,  # input_ids
                self._empty_sharding,  # position_ids
                es.extract_shardings(self.kv_pages, self.mesh),  # kv_pages
                self._empty_sharding,  # cache_metadata
                self._empty_sharding,  # logits_indices
                self._empty_sharding,  # sampling_params
                self._empty_sharding,  # rng_key
            ),
            out_shardings=(
                self._empty_sharding,
                es.extract_shardings(self.kv_pages, self.mesh),
                self._empty_sharding,
            ),
        )
        def _fn(
            graphdef,
            graphstate,
            graphother,
            input_ids: jax.Array,
            position_ids: jax.Array,
            kv_pages: PagesCache,
            cache_metadata: PagesMetadata,
            logits_indices: jax.Array,
            sampling_params: ModelRunnerSamplingMetadata,
            rng_key: jax.random.PRNGKey,
        ):
            model: EasyDeLBaseModule = nn.merge(graphdef, graphstate, graphother)
            with model.mesh:
                output = model(
                    input_ids=jnp.expand_dims(input_ids, 0),
                    position_ids=jnp.expand_dims(position_ids, 0),
                    past_key_values=kv_pages,
                    cache_metadata=cache_metadata,
                    apply_lm_head=False,
                )
                logits = model.apply_lm_head(output.last_hidden_state.squeeze(0)[logits_indices])
                keys = jax.random.split(rng_key, logits.shape[0] + 1)

                samples = jax.vmap(
                    sample_top_p_efficient,
                    in_axes=(0, 0, 0, 0, None),
                    out_axes=0,
                )(
                    logits,
                    sampling_params.top_p.astype(logits.dtype),
                    sampling_params.temperature.astype(logits.dtype),
                    keys[1:],
                    32,
                )
                return samples.reshape(-1, 1), output.past_key_values, keys[0]

        return _fn

    def compile_key(self, num_tokens: int, padded_num_reqs: int, compargs):
        """Compile model execution functions for specific input dimensions.

        Handles both AOT and JIT compilation modes based on use_aot_forward flag.
        For AOT mode (default), pre-compiles functions using JAX's lower/compile API.
        For JIT mode, executes functions once to trigger JIT compilation and caches
        the wrapped functions.

        Args:
            num_tokens: Number of tokens in the input batch.
            padded_num_reqs: Padded number of requests for batching.
            compargs: Compilation arguments for the model functions.
        """
        if self.use_fused_step:
            fused_key = (num_tokens, padded_num_reqs, "fused")
            if fused_key not in self._lowerd_history.keys():
                logger.debug(f"Compiling fused step function for key {fused_key}")
                lowered = self._fused_step_fn.lower(num_tokens, *compargs[2])
                compiled = lowered.compile()
                self._lowerd_history[fused_key] = compiled
        elif self.use_aot_forward:
            if self.use_combined_forward:
                logger.debug(f"Compiling combined forward function for key ({num_tokens}, {padded_num_reqs})")
                lowered = self._main_fn.lower(*compargs)
                compiled = lowered.compile()
                self._lowerd_history[(num_tokens, padded_num_reqs)] = compiled
            else:
                hskey = (num_tokens, padded_num_reqs, "hidden_states")
                tskey = (num_tokens, padded_num_reqs, "tokens")
                if hskey not in self._lowerd_history.keys():
                    logger.debug(f"Compiling hidden states function for key {hskey}")
                    hidden_states_lowered = self._compute_hidden_states_fn.lower(*compargs[0])
                    hidden_states_compiled = hidden_states_lowered.compile()
                    self._lowerd_history[hskey] = hidden_states_compiled
                if tskey not in self._lowerd_history.keys():
                    logger.debug(f"Compiling tokens function for key {tskey}")
                    tokens_lowered = self._compute_tokens_fn.lower(*compargs[1])
                    tokens_compiled = tokens_lowered.compile()
                    self._lowerd_history[tskey] = tokens_compiled
        else:
            if self.use_combined_forward:
                logger.debug(f"Compiling combined forward function for key ({num_tokens}, {padded_num_reqs})")
                _, self.kv_pages, _ = self._main_fn(*compargs)
                self._lowerd_history[(num_tokens, padded_num_reqs)] = self._main_fn
            else:
                hskey = (num_tokens, padded_num_reqs, "hidden_states")
                tskey = (num_tokens, padded_num_reqs, "tokens")
                if hskey not in self._lowerd_history.keys():
                    logger.debug(f"Compiling hidden states function for key {hskey}")
                    _, self.kv_pages = self._compute_hidden_states_fn(*compargs[0])
                    self._lowerd_history[hskey] = self._compute_hidden_states_fn
                if tskey not in self._lowerd_history.keys():
                    logger.debug(f"Compiling tokens function for key {tskey}")
                    _ = self._compute_tokens_fn(*compargs[1])
                    self._lowerd_history[tskey] = self._compute_tokens_fn

    def get_compiled_key(self, num_tokens: int, padded_num_reqs: int):
        """Retrieve pre-compiled functions for given input dimensions.

        Args:
            num_tokens: Number of tokens in the input batch.
            padded_num_reqs: Padded number of requests for batching.

        Returns:
            Compiled function(s) for the specified dimensions. Returns a single
            function for combined forward mode, fused step mode, or a tuple of
            (hidden_states_fn, tokens_fn) for separate mode.
        """
        if self.use_fused_step:
            fused_key = (num_tokens, padded_num_reqs, "fused")
            return self._lowerd_history[fused_key]
        elif self.use_combined_forward:
            return self._lowerd_history[(num_tokens, padded_num_reqs)]
        else:
            hskey = (num_tokens, padded_num_reqs, "hidden_states")
            tskey = (num_tokens, padded_num_reqs, "tokens")
            return self._lowerd_history[hskey], self._lowerd_history[tskey]

    def get_compile_configurations(
        self,
        kv_pages: PagesCache,
        rng_key: jax.random.PRNGKey,
        num_tokens: int,
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
        max_num_reqs: int,
        padded_num_reqs: int,
        metadata: PagesCacheMetaData,
    ) -> tuple:
        """Generate example arguments for function compilation.

        Creates mock input arguments with the correct shapes and types for
        compiling the execution functions. These arguments are used to trace
        through the functions during compilation.

        Args:
            kv_pages: KV cache pages to use in compilation.
            rng_key: Random key for sampling operations.
            num_tokens: Number of tokens in this configuration.
            num_reqs_max_model_len: Number of requests for max model length.
            max_pages_per_req: Maximum pages per request.
            max_num_reqs: Maximum number of requests.
            padded_num_reqs: Padded number of requests for this configuration.
            metadata: Pages cache metadata.

        Returns:
            A tuple of example arguments appropriate for the execution mode:
                - For fused mode: (None, None, fused_args)
                - For combined mode: Single tuple of arguments
                - For separate mode: (hidden_states_args, tokens_args)

        Note:
            The returned arguments contain zeros/ones as placeholder data since
            only shapes and types matter for compilation.
        """
        actual_num_reqs = min(num_tokens, num_reqs_max_model_len)
        padded_num_slices = metadata.get_padded_num_slices(num_tokens, max_num_reqs)
        query_lens = [1] * num_reqs_max_model_len

        if self.use_fused_step:
            from .sequence_buffer import SequenceBuffer

            temp_buffer = SequenceBuffer.create(
                max_num_reqs=max_num_reqs,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,  # Use the same value as eSurgeRunner
                vocab_size=self.model.config.get_text_config().vocab_size,
                page_sizes=[metadata.page_size],
            )

            # Get the DeviceSequenceState from the buffer
            dev_state = temp_buffer.to_device_state()

            max_padded_slices = metadata.get_padded_num_slices(self.max_model_len, max_num_reqs)

            fused_args = [
                self.graphdef,
                self.graphstate,
                self.graphother,
                dev_state,
                kv_pages,
                jnp.ones((max_num_reqs,), dtype=jnp.int32),  # scheduled_full
                jnp.full((max_num_reqs,), 10, dtype=jnp.int32),  # req_num_tokens_full
                jnp.ones((max_num_reqs,), dtype=bool),  # active_mask_full
                jnp.zeros((self.max_model_len,), dtype=jnp.int32),  # input_ids_buf
                jnp.zeros((self.max_model_len,), dtype=jnp.int32),  # position_ids_buf
                jnp.full((3, max_padded_slices), fill_value=SLOT_MAPPING_PADDING_VAL, dtype=jnp.int32),
                rng_key,
            ]

            example_args = [None, None, fused_args]  # (hidden_states_args, tokens_args, fused_args)
        elif self.use_combined_forward:
            example_args = (
                self.graphdef,
                self.graphstate,
                self.graphother,
                jnp.zeros((num_tokens,), dtype=jnp.int32),
                jnp.zeros(num_tokens, dtype=jnp.int32),
                kv_pages,
                PagesMetadata(
                    pages_tables=jnp.full(
                        (num_reqs_max_model_len, max_pages_per_req), fill_value=PAGE_TABLE_PADDING_VAL, dtype=jnp.int32
                    ),
                    context_lens=jnp.ones((num_reqs_max_model_len,), dtype=jnp.int32),
                    query_start_loc=jnp.cumsum(jnp.array([0, *query_lens], dtype=jnp.int32), axis=0, dtype=jnp.int32),
                    num_seqs=jnp.array([actual_num_reqs], dtype=jnp.int32),
                    slot_mapping=jnp.full((3, padded_num_slices), fill_value=SLOT_MAPPING_PADDING_VAL, dtype=jnp.int32),
                    num_kv_update_slices=jnp.array([padded_num_slices], dtype=jnp.int32),
                    num_slices_per_kv_cache_update_page=metadata.num_slices_per_kv_cache_update_page,
                    page_size=metadata.page_size,
                ),
                jnp.arange(padded_num_reqs, dtype=jnp.int32),
                ModelRunnerSamplingMetadata(
                    top_p=jnp.ones((padded_num_reqs,), dtype=jnp.float32),
                    temperature=jnp.ones((padded_num_reqs,), dtype=jnp.float32),
                    min_p=jnp.zeros((padded_num_reqs,), dtype=jnp.float32),
                    top_k=jnp.zeros((padded_num_reqs,), dtype=jnp.int32),
                ),
                rng_key,
            )
        else:
            example_args = (
                (
                    self.graphdef,
                    self.graphstate,
                    self.graphother,
                    jnp.zeros((num_tokens,), dtype=jnp.int32),
                    jnp.zeros(num_tokens, dtype=jnp.int32),
                    kv_pages,
                    PagesMetadata(
                        pages_tables=jnp.full(
                            (num_reqs_max_model_len, max_pages_per_req),
                            fill_value=PAGE_TABLE_PADDING_VAL,
                            dtype=jnp.int32,
                        ),
                        context_lens=jnp.ones((num_reqs_max_model_len,), dtype=jnp.int32),
                        query_start_loc=jnp.cumsum(
                            jnp.array([0, *query_lens], dtype=jnp.int32), axis=0, dtype=jnp.int32
                        ),
                        num_seqs=jnp.array([actual_num_reqs], dtype=jnp.int32),
                        slot_mapping=jnp.full(
                            (3, padded_num_slices), fill_value=SLOT_MAPPING_PADDING_VAL, dtype=jnp.int32
                        ),
                        num_kv_update_slices=jnp.array([padded_num_slices], dtype=jnp.int32),
                        num_slices_per_kv_cache_update_page=metadata.num_slices_per_kv_cache_update_page,
                        page_size=metadata.page_size,
                    ),
                ),
                (
                    self.graphdef,
                    self.graphstate,
                    self.graphother,
                    jnp.ones((num_tokens, self.model.config.get_text_config().hidden_size), self.model.dtype),
                    jnp.arange(padded_num_reqs, dtype=jnp.int32),
                    ModelRunnerSamplingMetadata(
                        top_p=jnp.ones((padded_num_reqs,), dtype=jnp.float32),
                        temperature=jnp.ones((padded_num_reqs,), dtype=jnp.float32),
                        min_p=jnp.zeros((padded_num_reqs,), dtype=jnp.float32),
                        top_k=jnp.zeros((padded_num_reqs,), dtype=jnp.int32),
                    ),
                    rng_key,
                ),
            )
        return example_args
