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

"""Execution manager for efficient model inference with precompiled fused step functions.

This module provides the ExecutionManager class that handles compilation and caching
of fused step execution functions for different batch sizes and token counts. It uses
AOT (Ahead-of-Time) compilation for optimal performance in production environments.

The manager uses a fused execution mode where a single function combines:
    - Input preparation (prepare_inputs)
    - Model forward pass
    - Token sampling
    - State updates (apply_token)

This provides maximum performance by minimizing host-device communication and
maximizing kernel fusion opportunities.

Example:
    >>> from easydel.inference.esurge.runners import ExecutionManager
    >>> executor = ExecutionManager(
    ...     model=my_model,
    ...     mesh=device_mesh,
    ...     kv_pages=cache_pages,
    ...     use_aot_forward=True,
    ... )
    >>> executor.compile(token_paddings, ...)
    >>> result = executor.execute(...)
"""

from __future__ import annotations

import typing
from collections import OrderedDict
from functools import partial

import jax
from eformer import escale as es
from eformer.loggings import ProgressLogger, get_logger
from flax import nnx as nn
from jax import numpy as jnp
from jax._src import pjit

from easydel.layers.caching import RaggedPagesCache, RaggedPagesCacheView, RaggedPagesMetadata
from easydel.utils import ejit

from ...sampling_funcs import sample_top_p_efficient
from ..page_table import PAGE_TABLE_PADDING_VAL, SLOT_MAPPING_PADDING_VAL
from .sequence_buffer import DeviceSequenceState

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
    """Manages precompiled fused step execution functions for efficient model inference.

    This class handles the compilation and caching of fused step execution functions
    for different token counts. The fused step combines input preparation, model
    forward pass, token sampling, and state updates into a single compiled function
    for maximum performance.

    The manager uses AOT (Ahead-of-Time) compilation, pre-compiling functions using
    JAX's lower/compile API for optimal performance in production environments.

    The manager pre-compiles functions for various token count configurations to avoid
    runtime compilation overhead, enabling seamless switching between different
    batch sizes during inference.

    Attributes:
        model: The EasyDeL model being managed.
        mesh: JAX sharding mesh for distributed execution.
        kv_pages: KV cache pages for attention.
        use_aot_forward: Whether to use AOT compilation (default: True).
        graphdef, graphstate, graphother: Split model components for JAX.
        _step_fn: The compiled fused step function.
        _lowerd_history: Cache of compiled functions.

    Example:
        >>> executor = ExecutionManager(
        ...     model=my_model,
        ...     mesh=device_mesh,
        ...     kv_pages=cache_pages,
        ...     use_aot_forward=True,
        ... )
        >>> executor.compile(token_paddings, ...)
        >>> result = executor.execute(...)
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

        self.rng_key = jax.random.PRNGKey(0)

        self._empty_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())

        self._step_fn: None | pjit.JitWrapped = None
        self._cache_capacity = 64
        self._lowerd_history = OrderedDict()

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

    def execute(
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

        """
        fn = self.get_compiled_key(num_tokens, padded_num_reqs)

        if self.use_aot_forward:
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
            hidden_states,
            logits,
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
        """Create the fused step function.

        Creates a single function that combines input preparation, model forward pass,
        token sampling, and state updates. This provides the best performance by
        minimizing host-device communication and maximizing kernel fusion.

        Returns:
            A callable that performs a complete inference step. The function is
            wrapped with ejit for efficient execution.

        """
        max_num_reqs = int(self.max_num_reqs)
        page_size = int(self.metadata.page_size)
        max_pages_per_req = int(self.metadata.max_num_pages_per_req)
        num_reqs_max_model_len = min(int(self.metadata.get_max_num_seqs()), max_num_reqs)
        slices_per_page = int(self.metadata.num_slices_per_kv_cache_update_page)
        page_table_pad = jnp.int32(PAGE_TABLE_PADDING_VAL)
        slot_mapping_pad = jnp.int32(SLOT_MAPPING_PADDING_VAL)
        max_num_tokens = int(self.max_num_tokens)
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
                self._empty_sharding,  # query_start_loc_buf
                self._empty_sharding,  # seq_lens_buf
                self._empty_sharding,  # pages_tables_buf
                self._empty_sharding,  # slot_mapping_buf
                self._empty_sharding,  # rng_key
                self._empty_sharding,  # out_tokens (full-size, masked)
                self._empty_sharding,  # valid_mask (full-size)
                self._empty_sharding,  # hidden_states
                self._empty_sharding,  # logits
            ),
        )
        def _fn(
            num_tokens_static: int,  # STATIC: padded_total bucket
            graphdef,
            graphstate,
            graphother,
            dev_state: DeviceSequenceState,
            kv_pages: RaggedPagesCache,
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

            tmp_logits = cum - 1
            mask_logits = i_reqs < padded_num_reqs
            logits_indices = jnp.where(mask_logits, tmp_logits, 0)

            input_ids_view = input_ids_buf[:num_tokens_static]
            position_ids_view = position_ids_buf[:num_tokens_static]

            model: EasyDeLBaseModule = nn.merge(graphdef, graphstate, graphother)
            with model.mesh:
                output = model(
                    input_ids=jnp.expand_dims(input_ids_view, 0),
                    position_ids=jnp.expand_dims(position_ids_view, 0),
                    past_key_values=kv_pages,
                    cache_metadata=RaggedPagesMetadata(
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

            is_all_greedy = jnp.all(temp <= 0.0)

            def do_greedy(_):
                return jnp.argmax(logits, axis=-1).astype(jnp.int32)

            def do_sample(_):
                B = logits.shape[0]
                row_keys = jax.vmap(lambda i: jax.random.fold_in(rng_key, i))(jnp.arange(B, dtype=jnp.int32))
                samples = jax.vmap(sample_top_p_efficient, in_axes=(0, 0, 0, 0, None), out_axes=0)(
                    logits, topp, temp, row_keys, 64
                )
                return samples.reshape(-1)

            sampled_flat = jax.lax.cond(is_all_greedy, do_greedy, do_sample, operand=None).reshape(-1)
            rng_key = jax.random.fold_in(rng_key, jnp.int32(num_tokens_static))

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
                rng_key,
                out_tokens_full,
                valid_mask_full,
                hs,
                logits,
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
        fused_key = (num_tokens, "fused")

        if fused_key not in self._lowerd_history:
            if self.use_aot_forward:
                compiled = self._step_fn.lower(num_tokens, *compargs).compile()
                self._cache_put(fused_key, compiled)
            else:
                partial_fn = partial(self._step_fn, num_tokens, self.graphdef)

                result = partial_fn(self.graphstate, self.graphother, *compargs[3:])
                (
                    _dev_state,
                    self.kv_pages,
                    _input_ids_buf,
                    _position_ids_buf,
                    _query_start_loc_buf,
                    _seq_lens_buf,
                    _pages_tables_buf,
                    _slot_mapping_buf,
                    _rng_key,
                    _out_tokens_full,
                    _valid_mask_full,
                    _hidden_states,
                    _logits,
                ) = result

                self._cache_put(fused_key, partial_fn)

    def get_compiled_key(self, num_tokens: int, padded_num_reqs: int):
        """Retrieve pre-compiled fused step function for given input dimensions.

        Args:
            num_tokens: Number of tokens in the input batch.
            padded_num_reqs: Padded number of requests for batching (unused in fused mode).

        Returns:
            Compiled fused step function for the specified number of tokens.
        """
        key = (num_tokens, "fused")
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
        num_tokens: int,  # unused - kept for API compatibility
        num_reqs_max_model_len: int,  # unused - kept for API compatibility
        max_pages_per_req: int,  # unused - kept for API compatibility
        max_num_reqs: int,
        padded_num_reqs: int,  # unused - kept for API compatibility
        metadata: RaggedPagesCacheView,
    ) -> tuple:
        """Generate example arguments for fused step function compilation.

        Creates mock input arguments with the correct shapes and types for
        compiling the fused step execution function. These arguments are used
        to trace through the function during compilation.

        Args:
            kv_pages: KV cache pages to use in compilation.
            rng_key: Random key for sampling operations.
            num_tokens: Unused in fused mode, kept for API compatibility.
            num_reqs_max_model_len: Unused in fused mode, kept for API compatibility.
            max_pages_per_req: Unused in fused mode, kept for API compatibility.
            max_num_reqs: Maximum number of requests.
            padded_num_reqs: Unused in fused mode, kept for API compatibility.
            metadata: Pages cache metadata.

        Returns:
            A tuple (None, None, fused_args) where fused_args contains the example
            arguments for the fused step function.

        Note:
            The returned arguments contain zeros/ones as placeholder data since
            only shapes and types matter for compilation.
        """
        from .sequence_buffer import SequenceBuffer

        temp_buffer = SequenceBuffer.create(
            max_num_reqs=max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            vocab_size=self.model.config.get_text_config().vocab_size,
            page_sizes=[metadata.page_size],
        )

        dev_state = temp_buffer.to_device_state()

        max_padded_slices = metadata.get_padded_num_slices(self.max_num_tokens, max_num_reqs)

        fused_args = [
            self.graphdef,
            self.graphstate,
            self.graphother,
            dev_state,
            kv_pages,
            jnp.ones((max_num_reqs,), dtype=jnp.int32),
            jnp.full((max_num_reqs,), 10, dtype=jnp.int32),
            jnp.ones((max_num_reqs,), dtype=bool),
            jnp.zeros((self.max_num_tokens,), dtype=jnp.int32),
            jnp.zeros((self.max_num_tokens,), dtype=jnp.int32),
            jnp.full((3, max_padded_slices), fill_value=SLOT_MAPPING_PADDING_VAL, dtype=jnp.int32),
            rng_key,
        ]
        return fused_args
