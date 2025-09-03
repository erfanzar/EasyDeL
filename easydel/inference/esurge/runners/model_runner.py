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
from dataclasses import replace

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

from ..metrics import get_metrics_collector
from ..outputs import ModelRunnerOutput
from ..page_table import PAGE_TABLE_PADDING_VAL, SLOT_MAPPING_PADDING_VAL
from ..scheduler import SchedulerOutput
from .execution_manager import ExecutionManager
from .sequence_buffer import SequenceBuffer
from .states import CachedRequestState

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule

logger = get_logger("eSurge")


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int, min_input_pad: int) -> int:
    """Calculate padded request count for compilation efficiency.

    Pads the number of requests to powers of 2 (up to 8) or the nearest
    power of 2 above 8. This reduces the number of unique compilations
    needed while maintaining good utilization.

    Args:
        x: Actual number of requests
        upper_limit: Maximum allowed requests

    Returns:
        int: Padded request count, capped at upper_limit

    Example:
        >>> _get_padded_num_reqs_with_upper_limit(3, 32)   # Returns 8
        >>> _get_padded_num_reqs_with_upper_limit(10, 32)  # Returns 16
        >>> _get_padded_num_reqs_with_upper_limit(20, 16)  # Returns 16
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
        max_model_len: int = 2**13,
        min_input_pad: int = 256,
        max_num_seqs: int = 16,
        verbose: bool = False,
    ):
        logger.debug(f"Initializing eSurgeRunner with {max_model_len=}, {max_num_seqs=}")
        logger.debug(f"Configuration: {hbm_utilization=}, {page_size=}")
        self.model = model
        self.metadata = model.create_paged_metadata(
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            max_model_length=max_model_len,
        )
        self.max_num_seqs = max_num_seqs
        self.max_num_reqs = max_num_seqs

        self.max_model_len = max_model_len
        self.min_input_pad = min(min_input_pad, max_num_seqs)

        self.page_size = int(self.metadata.page_size)
        self.max_pages_per_req = int(self.metadata.max_num_pages_per_req)

        self.num_tokens_paddings = self._get_token_paddings(
            min_token_size=16,
            max_token_size=self.max_model_len,
            padding_gap=0,
        )
        self.max_num_tokens = self.num_tokens_paddings[-1]

        logger.debug("Creating ExecutionManager and initializing pages cache")
        self.executor_manager = ExecutionManager(
            model=model,
            mesh=model.mesh,
            kv_pages=model.init_pages(self.metadata),
            use_combined_forward=False,
            use_aot_forward=True,
            use_fused_step=True,
            min_input_pad=self.min_input_pad,
            max_model_len=max_model_len,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            metadata=self.metadata,
        )
        self.log_it = logger.info if verbose else logger.debug
        self._setup_variables()
        logger.debug("eSurgeRunner initialization complete")

    @property
    def mesh(self):
        return self.model.mesh

    @property
    def _empty_sharding(self):
        return jax.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

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
        assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
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

    def _setup_variables(self):
        """Initialize internal variables and preallocate reusable buffers."""
        self.num_reqs_max_model_len = min(self.metadata.get_max_num_seqs(), self.max_num_reqs)
        self.num_reqs_most_model_len = self.num_reqs_max_model_len
        # num_tokens_paddings and max_num_tokens already calculated before ExecutionManager creation
        self.requests: dict[str, CachedRequestState] = {}
        logger.debug(f"Token padding sizes: {len(self.num_tokens_paddings)} levels, max={self.max_num_tokens}")

        logger.debug(
            f"Creating sequence buffer for max_num_reqs={self.max_num_reqs}, max_model_len={self.max_model_len}"
        )
        self.sequence_buffer = SequenceBuffer.create(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            vocab_size=self.model.config.get_text_config().vocab_size,
            page_sizes=[self.metadata.page_size],
        )

        self.arange = jnp.arange(self.max_num_tokens, dtype=jnp.int32)
        self.arange_np = jnp.arange(self.max_num_reqs, dtype=jnp.int32)

        self.input_ids_buf = jnp.zeros((self.max_num_tokens,), dtype=jnp.int32)
        self.position_ids_buf = jnp.zeros((self.max_num_tokens,), dtype=jnp.int32)
        self.query_start_loc_buf = jnp.zeros((self.max_num_reqs + 1,), dtype=jnp.int32)
        self.seq_lens_buf = jnp.zeros((self.max_num_reqs,), dtype=jnp.int32)

        self.pages_tables_buf = jnp.full(
            (self.num_reqs_max_model_len, self.max_pages_per_req),
            fill_value=PAGE_TABLE_PADDING_VAL,
            dtype=jnp.int32,
        )

        self.max_padded_slices = int(self.metadata.get_padded_num_slices(self.max_num_tokens, self.max_num_reqs))
        self.slot_mapping_buf = jnp.full(
            (3, self.max_padded_slices),
            fill_value=SLOT_MAPPING_PADDING_VAL,
            dtype=jnp.int32,
        )

        self.page_table_flat_buf = jnp.zeros(
            (self.num_reqs_max_model_len * self.max_pages_per_req,),
            dtype=jnp.int32,
        )

        self.slot_mapping_scratch_buf = jnp.zeros((self.max_padded_slices, 3), dtype=jnp.int32)
        self.num_tokens_paddings_arr = jnp.array(self.num_tokens_paddings, dtype=jnp.int32)

        # Pre-allocated buffers for fused execution to avoid repeated allocations
        self.scheduled_full_buf = jnp.zeros((self.max_num_reqs,), dtype=jnp.int32)
        self.req_num_tokens_full_buf = jnp.zeros((self.max_num_reqs,), dtype=jnp.int32)
        self.active_mask_full_buf = jnp.zeros((self.max_num_reqs,), dtype=bool)

        logger.debug(f"Allocated buffers: max_padded_slices={self.max_padded_slices}")

    def compile(self):
        """Compile the model for all token padding sizes."""
        logger.info("Starting eSurgeRunner compilation")
        logger.debug(
            f"Compiling for {len(self.num_tokens_paddings)} token padding sizes: {self.num_tokens_paddings[:5]}..."
            if len(self.num_tokens_paddings) > 5
            else f"Compiling for token padding sizes: {self.num_tokens_paddings}"
        )

        self.executor_manager.compile(
            num_tokens_paddings=self.num_tokens_paddings,
            num_reqs_max_model_len=self.num_reqs_max_model_len,
            max_pages_per_req=self.max_pages_per_req,
            max_num_reqs=self.max_num_reqs,
            metadata=self.metadata,
        )

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
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # 2) Remove finished from sequence buffer (functional)
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            self.sequence_buffer, req_index = self.sequence_buffer.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # 3) Remove unscheduled requests from buffer
        scheduled_req_ids = set(scheduler_output.num_scheduled_tokens.keys())
        cached_req_ids = set(self.sequence_buffer.req_id_to_index.keys())
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        for req_id in unscheduled_req_ids:
            self.sequence_buffer, req_index = self.sequence_buffer.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # 4) Add new requests to tracking
        req_ids_to_add: list[str] = []
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.sampling_params is not None, "Pooling not supported in TPU"
            req_id = new_req_data.req_id
            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=new_req_data.sampling_params,
                generator=None,
                page_ids=new_req_data.page_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
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
            batched_page_rows.append((req_index, new_page_ids))

        if upd_req_indices:
            idx_arr = jnp.array(upd_req_indices, dtype=jnp.int32)
            val_arr = jnp.array(upd_num_computed_vals, dtype=jnp.int32)
            new_num_computed = self.sequence_buffer.num_computed_tokens.at[idx_arr].set(val_arr)
            self.sequence_buffer = replace(self.sequence_buffer, num_computed_tokens=new_num_computed)

        if batched_page_rows:
            indices = [ix for ix, _ in batched_page_rows]
            pages_per_req = [ids for _, ids in batched_page_rows]
            new_pt = self.sequence_buffer.page_table.append_rows_batch(pages_per_req, indices)
            self.sequence_buffer = replace(self.sequence_buffer, page_table=new_pt)

        # 6) Add new / reinserted requests
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            reuse_index = removed_req_indices.pop() if removed_req_indices else None
            self.sequence_buffer = self.sequence_buffer.add_request(req_state, reuse_index)

        # 7) Condense to remove holes
        if removed_req_indices:
            self.sequence_buffer = self.sequence_buffer.condense(removed_req_indices)

        has_changes = len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0
        return has_changes

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
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
            ModelRunnerOutput: Contains:
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
        self._update_states(scheduler_output)
        updating_states_time = time.time() - updating_states_start

        if not scheduler_output.total_num_scheduled_tokens:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
                finished_sending=None,
                finished_recving=None,
                num_nans_in_logits=None,
            )

        start_index = 0
        total_exec_time = 0.0
        total_prep_time = 0.0
        total_sync_time = 0.0
        total_post_proc_time = 0.0

        # Result lists
        req_ids_all: list[str] = []
        sampled_token_ids_all: list[list[int]] = []

        # Initial device state conversion
        t_dev_state_start = time.time()
        dev_state = self.sequence_buffer.to_device_state()
        t_dev_state = time.time() - t_dev_state_start

        while start_index < self.sequence_buffer.num_reqs:
            # Window preparation timing
            t_prep_start = time.time()

            num_reqs_total = self.sequence_buffer.num_reqs
            scheduled_list: list[int] = []
            req_ids_window = []
            for i in range(start_index, min(num_reqs_total, start_index + self.num_reqs_max_model_len)):
                rid = self.sequence_buffer.req_ids[i]
                req_ids_window.append(rid)
                scheduled_list.append(int(scheduler_output.num_scheduled_tokens.get(rid, 0)) if rid is not None else 0)

            while scheduled_list and scheduled_list[-1] == 0:
                scheduled_list.pop()
                req_ids_window.pop()

            num_reqs = len(scheduled_list)
            if num_reqs == 0:
                break
            end_index = start_index + num_reqs

            # Host compute: num_tokens_static = smallest bucket >= total
            total_scheduled = sum(scheduled_list)
            idx = bisect_left(self.num_tokens_paddings, total_scheduled)
            if idx >= len(self.num_tokens_paddings):
                idx = len(self.num_tokens_paddings) - 1
            num_tokens_static = int(self.num_tokens_paddings[idx])

            # Array preparation using pre-allocated buffers
            # Only update the parts we need - no full reset needed since ExecutionManager only reads [:num_reqs]
            if num_reqs > 0:
                # Build arrays directly in numpy (faster for small arrays)
                scheduled_np = np.array(scheduled_list, dtype=np.int32)
                req_num_tokens_np = np.zeros(self.max_num_reqs, dtype=np.int32)
                active_mask_np = np.zeros(self.max_num_reqs, dtype=bool)

                for i, rid in enumerate(req_ids_window):
                    if rid is not None:
                        rs = self.requests.get(rid)
                        if rs:
                            req_num_tokens_np[i] = rs.num_tokens
                        active_mask_np[i] = True

                # Single conversion to JAX arrays
                self.scheduled_full_buf = jnp.asarray(scheduled_np, dtype=jnp.int32)
                if len(scheduled_np) < self.max_num_reqs:
                    # Pad with zeros
                    self.scheduled_full_buf = jnp.pad(
                        self.scheduled_full_buf,
                        (0, self.max_num_reqs - len(scheduled_np)),
                        constant_values=0,
                    )
                self.req_num_tokens_full_buf = jnp.asarray(req_num_tokens_np, dtype=jnp.int32)
                self.active_mask_full_buf = jnp.asarray(active_mask_np, dtype=bool)

            # Calculate padded_num_reqs
            nr_safe = max(num_reqs, 1)
            next_pow2 = 1 << (nr_safe - 1).bit_length()
            padded_num_reqs = min(self.min_input_pad if num_reqs <= self.min_input_pad else next_pow2, self.max_num_reqs)

            t_prep = time.time() - t_prep_start
            total_prep_time += t_prep

            # Execute model
            exec_start = time.time()
            (
                dev_state,
                out_tokens_win,
                valid_mask_win,
                self.input_ids_buf,
                self.position_ids_buf,
                self.query_start_loc_buf,
                self.seq_lens_buf,
                self.pages_tables_buf,
                self.slot_mapping_buf,
            ) = self.executor_manager.execute_fused(
                num_tokens=num_tokens_static,
                dev_state=dev_state,
                scheduled_full=self.scheduled_full_buf,
                req_num_tokens_full=self.req_num_tokens_full_buf,
                active_mask_full=self.active_mask_full_buf,
                input_ids_buf=self.input_ids_buf,
                position_ids_buf=self.position_ids_buf,
                slot_mapping_buf=self.slot_mapping_buf,
                padded_num_reqs=padded_num_reqs,
            )
            t_exec = time.time() - exec_start
            total_exec_time += t_exec

            # Async copy
            out_tokens_win.copy_to_host_async()
            valid_mask_win.copy_to_host_async()

            # Sequence buffer update timing
            sq_utime = time.time()
            self.sequence_buffer = self.sequence_buffer.from_device_state(dev_state)
            sq_utime_took = time.time() - sq_utime
            total_sync_time += sq_utime_took

            # Post-processing - highly optimized
            up_wtime = time.time()

            # Direct boolean array indexing (fastest)
            for i, rid in enumerate(req_ids_window):
                if rid is None:
                    continue
                req_ids_all.append(rid)

                if valid_mask_win[i]:
                    # Direct extraction without conversion overhead
                    tid = int(out_tokens_win[i])
                    sampled_token_ids_all.append([tid])
                    # Only lookup if we need to update
                    if rid in self.requests:
                        self.requests[rid].output_token_ids.append(tid)
                else:
                    sampled_token_ids_all.append([])

            up_wtime_took = time.time() - up_wtime
            total_post_proc_time += up_wtime_took

            start_index = end_index

        # kv_pages and rng_key are already updated inside executor_manager

        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.record_runner_metrics(
                execution_time=time.time() - execution_start_time,
                batch_size=len(req_ids_all),
                num_tokens=scheduler_output.total_num_scheduled_tokens,
            )

        total_time = time.time() - execution_start_time
        self.log_it(
            f"[fused] exec={total_exec_time:.3f}s "
            f"prep={total_prep_time:.3f}s "
            f"sync={total_sync_time:.3f}s "
            f"post={total_post_proc_time:.3f}s "
            f"init_dev={t_dev_state:.3f}s "
            f"upd_states={updating_states_time:.3f}s "
            f"total={total_time:.3f}s"
        )

        return ModelRunnerOutput(
            req_ids=req_ids_all,
            req_id_to_index=self.sequence_buffer.req_id_to_index,
            sampled_token_ids=sampled_token_ids_all,
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={rid: None for rid in req_ids_all},
            finished_sending=None,
            finished_recving=None,
        )
