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

from __future__ import annotations

import queue
import time
import traceback
import typing as tp
from functools import cached_property

import jax
import numpy as np
from jax import numpy as jnp

from easydel.inference.vsurge.scheduler.page_scheduler import DecodeScheduleInfo
from easydel.layers.caching import PagesMetadata
from easydel.utils.helpers import get_logger

from ...sampling_params import JitableSamplingParams
from ..scheduler import BatchConfig, PagedScheduler, PrefillScheduleInfo, RequestPriority, SchedulePolicy, Scheduler
from ..utils import (
    ActiveRequest,
    GenerationState,
    MetricsRecorder,
    ResultTokens,
    ReturnSample,
    SafeThread,
    pad_tokens,
    process_result_tokens,
)
from .engine import vEngine

if tp.TYPE_CHECKING:
    from easydel.infra.utils import ProcessingClassType

logger = get_logger("vSurge-vDriver")


class vDriver:
    """
    Drives the vEngine for prefill and decode operations, managing request flow.

    The `vDriver` orchestrates the entire inference pipeline, including request
    submission, prefilling, KV cache transfer, decoding, and detokenization.
    It uses a unified inference thread and scheduler to manage these stages
    efficiently. It also incorporates a `MetricsRecorder` to track operational
    statistics.

    Attributes:
        _engine (vEngine): The underlying engine performing model computations.
        _detokenize_backlog (queue.Queue): Queue for results awaiting detokenization.
        _live_requests (dict): Tracks requests currently active in decode slots.
        _interleaved_mode (bool): If True, prioritizes new requests for lower latency.
        _slot_clear_steps (int): Interval for clearing unused resources in decode state.
        _detokenizing_blocks (int): Max size for the detokenize backlog.
        metrics_recorder (MetricsRecorder): Instance for recording metrics.
        log (function): Logger instance for outputting messages.
        live (bool): Flag indicating if the driver's worker threads are active.
        _pause (bool): Flag indicating if the driver is paused.
        _all_threads (list[SafeThread]): list of all managed background threads.
        _metrics_thread (SafeThread | None): Thread for monitoring and logging metrics.
    """

    _engine: vEngine
    _detokenize_backlog: queue.Queue[tp.Any]

    def __init__(
        self,
        engine: vEngine,
        schedule_policy: SchedulePolicy = SchedulePolicy.ADAPTIVE,
        batch_config: BatchConfig | None = None,
        detokenizing_blocks: int = 8,
        max_prefill_chunk_size: int = 512,
        slot_clear_steps: int = 512,
        verbose: bool = True,
        metrics_log_interval_sec: float = 10.0,
        interleaved_mode: bool = False,
    ):
        """
        Initialize the vDriver.

        Args:
            engine: The vEngine instance to drive
            schedule_policy: Scheduling policy (ONLINE, OFFLINE, ADAPTIVE)
            batch_config: Configuration for request batching (SimpleScheduler only)
            detokenizing_blocks: Size of detokenization queue
            slot_clear_steps: Steps between resource cleanup
            verbose: Enable verbose logging
            metrics_log_interval_sec: Interval for metrics logging
            interleaved_mode: If True, skip KV cache transfer step
        """
        self._engine = engine
        self._detokenizing_blocks = detokenizing_blocks
        self._slot_clear_steps = slot_clear_steps
        self._interleaved_mode = interleaved_mode
        self._max_prefill_chunk_size = max_prefill_chunk_size
        self._pause = False

        if self.is_paged_runtime:
            self.scheduler = PagedScheduler(
                metadata=engine.page_metadata,
                max_batch_size=engine.max_concurrent_decodes,
                max_prefill_batch_size=engine.max_concurrent_prefill,
                max_prefill_chunk_size=max_prefill_chunk_size,
                schedule_policy=schedule_policy,
            )
        else:
            self.scheduler = Scheduler(
                max_batch_size=engine.max_concurrent_decodes,
                max_prefill_batch_size=engine.max_concurrent_prefill,
                schedule_policy=schedule_policy,
                batch_config=batch_config,
            )

        self._detokenize_backlog = queue.Queue(maxsize=self._detokenizing_blocks if self._detokenizing_blocks > 0 else 0)

        self._live_requests: dict[int, ActiveRequest | None] = {i: None for i in range(engine.max_concurrent_decodes)}

        self.metrics_recorder = MetricsRecorder(metrics_log_interval_sec)
        self._request_counter = 0

        self.log = logger.info if verbose else logger.debug

        self.live = False
        self._all_threads: list[SafeThread] = []
        self._metrics_thread: SafeThread | None = None

        if self.is_paged_runtime:
            self._setup_pages_requirements()

    def _setup_pages_requirements(self):
        """Setup requirements for paged attention using PagesMetadata"""
        engine = self.engine
        metadata = engine.page_metadata
        self.pages_metadata = PagesMetadata.create_empty(
            num_tokens=engine.max_prefill_length,
            max_num_reqs=engine.max_concurrent_decodes,
            max_blocks=metadata.pages_per_sequence,
            page_size=metadata.page_size,
        )
        self.log(
            f"Initialized PagesMetadata with page_size={metadata.page_size}, max_blocks={metadata.pages_per_sequence}"
        )

    @cached_property
    def is_paged_runtime(self) -> bool:
        return self.engine.is_paged_runtime

    @property
    def engine(self) -> vEngine:
        return self._engine

    @property
    def max_prefill_chunk_size(self) -> int:
        return self._max_prefill_chunk_size

    @property
    def driver_name(self) -> str:
        """
        Returns a standardized name for the driver and its model.

        The name typically includes the model type and size.

        Returns:
            str: The driver name.
        """
        return self._get_model_name(self._engine.model)

    @property
    def processor(self) -> ProcessingClassType:
        """
        Returns the tokenizer/processor associated with the engine.

        Returns:
            ProcessingClassType: The processor instance.
        """
        return self._engine.processor

    def _calculate_model_size(self, graphstate) -> str:
        """
        Calculates the model size in billions of parameters.

        Args:
            graphstate: The model's graph state containing parameters.

        Returns:
            str: Formatted model size (e.g., "7.00b") or "unknown".
        """
        try:
            num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
            size_in_billions = num_params / 1e9
            return f"{size_in_billions:.2f}"
        except Exception:
            return "unknown"

    def _get_model_name(self, model) -> str:
        """
        Generates a model name string including type and size.

        Args:
            model: The model object.

        Returns:
            str: The generated model name (e.g., "llama-7.00b").
        """
        model_type = self._get_model_type(model)
        model_size = self._calculate_model_size(model.graphstate)
        return f"{model_type}-{model_size}b"

    def _get_model_type(self, model) -> str:
        """
        Extracts the model type from the model's configuration.

        Args:
            model: The model object.

        Returns:
            str: The model type in lowercase, or "unknown".
        """
        return getattr(model.config, "model_type", "unknown").lower()

    def compile(self):
        """
        Compiles the prefill, insert, and decode functions of the engine.

        This method runs dummy inputs through the core engine operations to
        trigger JAX JIT compilation, ensuring subsequent calls are fast.
        Exits on compilation failure.
        """
        engine = self._engine
        try:
            self.log("Starting engine compilation...")
            decode_state = engine.init_decode_state()
            for length in engine.prefill_lengths:
                padded_tokens = padded_valids = jnp.ones((1, length), "i4")
                self.log(f"Compiling prefill/insert length={length}")
                state_new, _ = engine.prefill(
                    graphstate=engine.graphstate,
                    graphothers=engine.graphothers,
                    tokens=padded_tokens,
                    valids=padded_valids,
                    true_length=0,
                    sampling_params=JitableSamplingParams.init_empty(1).view_1d(),
                    rngs=engine.prng_key,
                    cache=None,
                    cache_metadata=None,
                    slot=0,
                )
                decode_state = engine.insert(state_new, decode_state, 0)

            self.log("Compiling decode")
            decode_state = engine.free_state_resources([0], decode_state)
            decode_state, _ = engine.decode(
                graphstate=engine.graphstate,
                graphothers=engine.graphothers,
                state=decode_state,
                cache_metadata=None,
                rngs=engine.prng_key,
                slot=0,
            )
            engine.free_resource(0)
            del decode_state
            self.log("Engine compilation completed successfully")
        except Exception as e:
            self.log(f"Compilation failed: {e}")
            traceback.print_exc()
            self.stop()
            exit(1)

    def get_total_concurrent_requests(self) -> int:
        """
        Returns the total number of concurrent decode requests the engine can handle.

        Returns:
            int: Maximum concurrent decodes supported by the engine.
        """
        return self._engine.max_concurrent_decodes  # Fixed: removed 'total_' prefix

    def _detokenize_action_thread(self):
        """
        Background thread action for detokenizing results and returning samples.

        This thread consumes items from the `_detokenize_backlog`. It handles
        three types of items:
        1. Initial tokens from a prefill operation.
        2. Subsequent tokens from decode steps.
        3. New `ActiveRequest` objects to start tracking in `_live_requests`.

        It processes tokens, checks for completion (EOS, stop sequences),
        and enqueues `ReturnSample` objects to the request's return channel.
        """
        engine = self._engine
        processor = engine.processor
        while self.live:
            try:
                data: tp.Any = self._detokenize_backlog.get(block=True, timeout=0.1)
            except queue.Empty:
                if not self.live:
                    break
                continue
            if data is None:
                break

            if isinstance(data[0], ResultTokens):
                request_first_token, request, prefill_dequeue_time = data
                request_first_token = request_first_token.convert_to_numpy()
                results_base, complete, num_valid_tokens_list = process_result_tokens(
                    processor=processor,
                    slot=0,
                    slot_max_length=request.sampling_params.max_tokens,
                    result_tokens=request_first_token,
                    eos_token_id=engine.eos_token_ids,
                    is_client_side_tokenization=request.is_client_side_tokenization,
                    complete=request.complete,
                    ignore_eos=request.sampling_params.ignore_eos,
                )
                request.complete = complete
                final_results = []
                for res_base, num_valid in zip(results_base, num_valid_tokens_list, strict=False):
                    request.accumulated_text = res_base.text
                    request.total_generated_tokens += num_valid
                    final_results.append(
                        ReturnSample(
                            text=res_base.text,
                            token_ids=res_base.token_ids,
                            time_spent_computing=0.0,
                            accumulated_text=request.accumulated_text,
                            tokens_per_second=0.0,
                            num_generated_tokens=request.total_generated_tokens,
                        )
                    )
                if request.return_channel:
                    request.enqueue_samples(final_results)

                first_token_return_time = (time.perf_counter() - prefill_dequeue_time) * 1000
                self.metrics_recorder.record_ttft(first_token_return_time)
                self.log(f"TTFT: {first_token_return_time:.2f}ms for request {request.id}")

            elif len(data) == 2 and isinstance(data[1], ResultTokens):
                _, result_tokens = data
                result_tokens = result_tokens.convert_to_numpy()

                for slot, request_obj in list(self._live_requests.items()):
                    if request_obj is not None:
                        request: ActiveRequest = request_obj
                        if request.decode_start_time is None:
                            request.decode_start_time = time.perf_counter()
                        results_base, complete, num_valid_tokens_list = process_result_tokens(
                            processor=processor,
                            slot=slot,
                            slot_max_length=request.sampling_params.max_tokens,
                            result_tokens=result_tokens,
                            eos_token_id=engine.eos_token_ids,
                            is_client_side_tokenization=request.is_client_side_tokenization,
                            complete=request.complete,
                            ignore_eos=request.sampling_params.ignore_eos,
                        )
                        request.complete = complete
                        elapsed_time = time.perf_counter() - request.decode_start_time
                        final_step_results = []
                        for res_base, num_valid in zip(results_base, num_valid_tokens_list, strict=False):
                            if len(res_base.text) > 0:
                                if isinstance(request.accumulated_text, list):
                                    for idx, (accum, res) in enumerate(
                                        zip(request.accumulated_text, res_base.text, strict=False)
                                    ):
                                        request.accumulated_text[idx] = accum + res
                                else:
                                    request.accumulated_text = request.accumulated_text + res_base.text
                            if request.sampling_params.stop is not None:
                                for stop_sign in request.sampling_params.stop:
                                    if isinstance(request.accumulated_text, list):
                                        for idx, accum in enumerate(request.accumulated_text):
                                            if stop_sign in accum:
                                                request.complete[idx] = True
                                    else:
                                        if stop_sign in request.accumulated_text:
                                            request.complete = np.ones_like(request.complete, dtype=bool)
                            request.total_generated_tokens += num_valid
                            tps = request.total_generated_tokens / elapsed_time if elapsed_time > 1e-6 else 0.0
                            final_step_results.append(
                                ReturnSample(
                                    text=res_base.text,
                                    token_ids=res_base.token_ids,
                                    time_spent_computing=elapsed_time,
                                    accumulated_text=request.accumulated_text,
                                    tokens_per_second=tps,
                                    num_generated_tokens=request.total_generated_tokens,
                                )
                            )
                        if request.return_channel:
                            request.enqueue_samples(final_step_results)

                        if request.complete.all():
                            if request.return_channel:
                                request.return_channel.close()
                            self.log(
                                f"Request {request.id} in slot {slot} completed. "
                                f"Generated {request.total_generated_tokens} tokens in {elapsed_time:.2f}s "
                                f"({tps:.2f} tokens/s)"
                            )
                            self.metrics_recorder.increment_completed_requests()
                            self._live_requests[slot] = None
                            # Notify scheduler that request is complete
                            self.scheduler.complete_request(request.id)
                            engine.free_resource(slot)
            elif len(data) == 2 and isinstance(data[1], ActiveRequest):
                slot, active_request = data
                self.log(f"Tracking new active request {active_request.id} in slot {slot}.")
                self._live_requests[slot] = active_request
            else:
                self.log(
                    f"Warning: Unknown data type received in detokenize backlog: {type(data[0]) if data else 'None'}"
                )

    def _unified_inference_thread(self):
        """Unified inference thread that handles both paged and non-paged modes"""
        engine = self._engine
        processor = engine.processor
        decode_state = engine.init_decode_state()
        generate_timestep = 0

        # Initialize prefill accumulator for paged mode
        prefill_accumulator = {} if self.is_paged_runtime else None

        # Initialize page manager if needed
        if self.is_paged_runtime and not hasattr(self, "page_manager"):
            self.page_manager = self.scheduler.page_manager

        while self.live:
            try:
                if self.is_paged_runtime:
                    schedule_result = self.scheduler.schedule()

                    # Update page manager
                    if schedule_result.updated_page_manager:
                        self.page_manager = schedule_result.updated_page_manager

                    # Process prefills
                    if schedule_result.should_prefill:
                        decode_state = self._process_paged_prefill(
                            engine,
                            processor,
                            schedule_result.prefill_infos,
                            prefill_accumulator,
                            decode_state,
                            generate_timestep,
                        )

                    # Process decodes
                    if schedule_result.should_decode and schedule_result.decode_info:
                        decode_state, sampled_tokens = self._process_paged_decode(
                            engine,
                            decode_state,
                            schedule_result.decode_info,
                            generate_timestep,
                        )
                        generate_timestep += 1
                else:
                    # Simple scheduler path remains the same
                    simple_result = self.scheduler.schedule()

                    if simple_result.should_prefill:
                        decode_state = self._process_fifo_prefill(
                            engine,
                            processor,
                            simple_result.prefill_requests,
                            decode_state,
                            generate_timestep,
                        )

                    if simple_result.should_decode:
                        decode_state, sampled_tokens = self._process_fifo_decode(
                            engine,
                            decode_state,
                            simple_result.decode_slots,
                            generate_timestep,
                        )
                        generate_timestep += 1

                # Periodic cleanup
                if self._slot_clear_steps and (generate_timestep % self._slot_clear_steps) == 0:
                    decode_state = self._perform_cleanup(engine, decode_state)

                # Sleep if nothing to do
                should_sleep = True
                if self.is_paged_runtime:
                    should_sleep = not (schedule_result.should_prefill or schedule_result.should_decode)
                else:
                    should_sleep = not (simple_result.should_prefill or simple_result.should_decode)

                if should_sleep:
                    time.sleep(0.001)

            except Exception as e:
                self.log(f"Fatal error in inference thread: {e}")
                traceback.print_exc()
                break

    def _update_periodic_metrics(self):
        """Update metrics that are tracked periodically"""
        if hasattr(self.scheduler, "prefill_queue"):
            self.metrics_recorder.update_queue_size("prefill_pending", self.scheduler.prefill_queue.qsize())

        self.metrics_recorder.update_queue_size("detokenize_backlog", self._detokenize_backlog.qsize())

        active_count = sum(1 for req in self._live_requests.values() if req is not None)
        self.metrics_recorder.set_active_requests_count(active_count)

    def _perform_cleanup(self, engine, decode_state):
        """Perform periodic cleanup of unused resources"""
        self.log("Performing periodic cleanup")

        inactive_slots = [slot for slot, req in self._live_requests.items() if req is None]

        if inactive_slots:
            self.log(f"Clearing {len(inactive_slots)} unused slots: {inactive_slots}")
            decode_state = engine.free_state_resources(inactive_slots, decode_state)

        return decode_state

    def _process_fifo_prefill(
        self,
        engine,
        processor,
        prefill_requests: list[tuple[int, ActiveRequest]],
        decode_state,
        generate_timestep: int,
    ):
        """Process prefill for simple scheduler (entire prompts at once)"""
        prefill_start_batch = time.perf_counter()

        for slot, request in prefill_requests:
            try:
                request.metadata.prefill_dequeue_time = time.perf_counter()

                padded_tokens, padded_valids, true_length = self._process_prefill_content(
                    request,
                    processor,
                    engine.max_prefill_length,
                    engine.prefill_lengths,
                    engine.pad_token_id,
                )

                self.log(f"Prefilling request {request.id} in slot {slot} (simple mode)")

                prefill_start = time.perf_counter()

                prefill_result, first_token = engine.prefill(
                    graphstate=engine.graphstate,
                    graphothers=engine.graphothers,
                    tokens=padded_tokens,
                    valids=padded_valids,
                    true_length=true_length,
                    sampling_params=request.sampling_params.make_jitable().view_1d(),
                    cache=None,  # No cache for simple mode
                    cache_metadata=None,  # No metadata for simple mode
                    rngs=engine.prng_key,
                    slot=slot,
                )

                prefill_duration_ms = (time.perf_counter() - prefill_start) * 1000
                self.metrics_recorder.record_prefill_op_time(prefill_duration_ms)

                if not self._interleaved_mode:
                    transfer_start = time.perf_counter()
                    dst_sharding = engine.get_prefix_destination_sharding()
                    prefill_result = jax.device_put(prefill_result, dst_sharding)
                    jax.block_until_ready(prefill_result)
                    transfer_duration_ms = (time.perf_counter() - transfer_start) * 1000
                    self.metrics_recorder.record_transfer_op_time(transfer_duration_ms)

                insert_start = time.perf_counter()
                decode_state = engine.insert(prefix=prefill_result, decode_state=decode_state, slot=slot)
                insert_duration_ms = (time.perf_counter() - insert_start) * 1000
                self.metrics_recorder.record_insert_op_time(insert_duration_ms)

                request.prefill_result = prefill_result
                request.complete = np.zeros((engine.samples_per_slot,), "b1")
                request.generate_timestep_added = generate_timestep
                request.decode_start_time = None

                self._detokenize_backlog.put(
                    (first_token, request, request.metadata.prefill_dequeue_time),
                    block=True,
                )

                self._detokenize_backlog.put((slot, request), block=True)
                self._live_requests[slot] = request

                del prefill_result

            except Exception as e:
                self.log(f"Error processing prefill for request {request.id}: {e}")
                if request.return_channel:
                    request.return_channel.set_exception(e)
                self.scheduler.complete_request(request.id)

        prefill_batch_duration = (time.perf_counter() - prefill_start_batch) * 1000
        self.log(
            f"Simple prefill batch completed: {len(prefill_requests)} requests, "
            f"total time: {prefill_batch_duration:.2f}ms"
        )
        return decode_state

    def _process_fifo_decode(self, engine, decode_state, decode_slots: list[int], generate_timestep: int):
        """Process decode for simple scheduler"""
        decode_start = time.perf_counter()
        active_in_decode_slots = sum(
            1 for slot in decode_slots if slot < len(self._live_requests) and self._live_requests[slot] is not None
        )

        try:
            decode_state, sampled_tokens = engine.decode(
                graphstate=engine.graphstate,
                graphothers=engine.graphothers,
                state=decode_state,
                cache_metadata=None,
                rngs=engine.prng_key,
                slot=0,
            )

            decode_duration_ms = (time.perf_counter() - decode_start) * 1000
            self.metrics_recorder.record_decode_op_time(decode_duration_ms)

            sampled_tokens.copy_to_host_async()
            self._detokenize_backlog.put((generate_timestep, sampled_tokens), block=True)

            total_cycle_time = (time.perf_counter() - decode_start) * 1000
            self.log(
                f"Simple decode completed - "
                f"DecodeOp: {decode_duration_ms:.2f}ms, "
                f"TotalCycle: {total_cycle_time:.2f}ms, "
                f"ActiveSlots: {active_in_decode_slots}/{engine.max_concurrent_decodes}, "
                f"step: {generate_timestep}"
            )

            return decode_state, sampled_tokens

        except Exception as e:
            self.log(f"Error in simple decode step: {e}")
            traceback.print_exc()
            return decode_state, None

    def _process_paged_prefill(
        self,
        engine,
        processor,
        prefill_infos: list[PrefillScheduleInfo],
        prefill_accumulator: dict,
        decode_state,
        generate_timestep: int,
    ):
        """Process prefill for paged scheduler (chunked)"""
        prefill_start_batch = time.perf_counter()

        for info in prefill_infos:
            slot = info.slot
            request = info.request
            page_info = info.page_batch_info

            try:
                # Initialize request state if first chunk
                if not hasattr(request, "_tokenized_ids"):
                    if isinstance(request.prefill_content, str):
                        content = processor(
                            text=request.prefill_content, return_tensors="np", return_attention_mask=True
                        )
                        request._tokenized_ids = content["input_ids"][0].tolist()
                        request._attention_mask = content["attention_mask"][0]
                    else:
                        request._tokenized_ids = list(request.prefill_content)
                        request._attention_mask = [1] * len(request._tokenized_ids)

                # Get the actual chunk tokens
                chunk_start = request.prefill_tokens_processed - len(info.token_ids)
                chunk_end = request.prefill_tokens_processed
                actual_tokens = jnp.array(request._tokenized_ids[chunk_start:chunk_end])
                actual_mask = jnp.array(request._attention_mask[chunk_start:chunk_end])

                # Get existing cache for this slot if continuing
                existing_cache = None
                if slot in prefill_accumulator:
                    existing_cache = prefill_accumulator[slot]["cache"]

                # Process the chunk
                prefill_result, chunk_output = self._process_prefill_chunk(
                    engine=engine,
                    tokens=actual_tokens,
                    mask=actual_mask,
                    request=request,
                    slot=slot,
                    page_info=page_info,
                    decode_state=decode_state,
                    existing_cache=existing_cache,
                    is_first_chunk=(chunk_start == 0),
                )

                # Store accumulated state
                if slot not in prefill_accumulator:
                    prefill_accumulator[slot] = {}

                prefill_accumulator[slot]["cache"] = prefill_result.cache
                prefill_accumulator[slot]["last_output"] = chunk_output
                prefill_accumulator[slot]["request"] = request

                # Handle completion
                if request.is_prefill_complete:
                    self._finalize_prefill(
                        engine=engine,
                        request=request,
                        slot=slot,
                        prefill_result=prefill_result,
                        decode_state=decode_state,
                        generate_timestep=generate_timestep,
                        first_token=prefill_accumulator[slot].get("last_output"),
                    )
                    # Clean up accumulator
                    prefill_accumulator.pop(slot, None)

            except Exception as e:
                self.log(f"Error processing paged prefill chunk for request {request.id}: {e}")
                traceback.print_exc()
                if request.return_channel:
                    request.return_channel.set_exception(e)
                self.scheduler.complete_request(request.id)
                prefill_accumulator.pop(slot, None)

        prefill_batch_duration = (time.perf_counter() - prefill_start_batch) * 1000
        self.log(
            f"Paged prefill batch completed: {len(prefill_infos)} chunks, total time: {prefill_batch_duration:.2f}ms"
        )

    def _process_prefill_chunk(
        self,
        engine,
        tokens: jnp.ndarray,
        mask: jnp.ndarray,
        request: ActiveRequest,
        slot: int,
        page_info: PagesMetadata,
        decode_state: GenerationState,
        existing_cache=None,
        is_first_chunk: bool = False,
    ) -> tuple:
        """Process a single prefill chunk"""
        # Prepare tokens with proper padding
        padded_tokens, padded_valids, true_length = pad_tokens(
            tokens=tokens[None, :],
            valids=mask[None, :],
            pad_token_id=engine.pad_token_id,
            max_prefill_length=self.max_prefill_chunk_size,
            prefill_lengths=engine.prefill_lengths,
            right_padding=False,
        )

        # Use existing cache or get from decode_state
        cache_to_use = existing_cache if existing_cache is not None else decode_state.cache

        # Call prefill with proper cache
        prefill_result, chunk_output = engine.prefill(
            graphstate=engine.graphstate,
            graphothers=engine.graphothers,
            tokens=padded_tokens,
            valids=padded_valids,
            true_length=true_length,
            sampling_params=request.sampling_params.make_jitable().view_1d(),
            cache=cache_to_use,
            cache_metadata=page_info,
            rngs=engine.prng_key,
            slot=slot,
        )

        if is_first_chunk:
            request.metadata.prefill_dequeue_time = time.perf_counter()
            self._detokenize_backlog.put(
                (chunk_output, request, request.metadata.prefill_dequeue_time),
                block=True,
            )

        return prefill_result, chunk_output

    def _process_prefill_chunk(
        self,
        engine,
        tokens: jnp.ndarray,
        mask: jnp.ndarray,
        request: ActiveRequest,
        slot: int,
        page_info: PagesMetadata,
        decode_state: GenerationState,
        existing_state=None,
        is_first_chunk: bool = False,
    ) -> tuple:
        """Process a single prefill chunk"""
        padded_tokens, padded_valids, true_length = pad_tokens(
            tokens=tokens[None, :],
            valids=mask[None, :],
            pad_token_id=engine.pad_token_id,
            max_prefill_length=self.max_prefill_chunk_size,
            prefill_lengths=engine.prefill_lengths,
            right_padding=False,
        )

        prefill_result, chunk_output = engine.prefill(
            graphstate=engine.graphstate,
            graphothers=engine.graphothers,
            tokens=padded_tokens,
            valids=padded_valids,
            true_length=true_length,
            sampling_params=request.sampling_params.make_jitable().view_1d(),
            cache=decode_state.cache,
            cache_metadata=page_info,
            rngs=engine.prng_key,
            slot=slot,
        )

        if is_first_chunk:
            request.metadata.prefill_dequeue_time = time.perf_counter()
            self._detokenize_backlog.put(
                (chunk_output, request, request.metadata.prefill_dequeue_time),
                block=True,
            )

        return prefill_result, chunk_output

    def _finalize_prefill(
        self,
        engine,
        request: ActiveRequest,
        slot: int,
        prefill_result,
        decode_state,
        generate_timestep: int,
        first_token=None,
    ):
        """Finalize prefill and prepare for decode"""
        if not self._interleaved_mode:
            transfer_start = time.perf_counter()
            dst_sharding = engine.get_prefix_destination_sharding()
            prefill_result = jax.device_put(prefill_result, dst_sharding)
            jax.block_until_ready(prefill_result)
            transfer_duration_ms = (time.perf_counter() - transfer_start) * 1000
            self.metrics_recorder.record_transfer_op_time(transfer_duration_ms)

        # Insert into decode state
        insert_start = time.perf_counter()
        decode_state = engine.insert(prefix=prefill_result, decode_state=decode_state, slot=slot)
        insert_duration_ms = (time.perf_counter() - insert_start) * 1000
        self.metrics_recorder.record_insert_op_time(insert_duration_ms)

        # Setup request state
        request.prefill_result = prefill_result
        request.complete = np.zeros((engine.samples_per_slot,), "b1")
        request.generate_timestep_added = generate_timestep
        request.decode_start_time = None

        # Track in live requests
        self._detokenize_backlog.put((slot, request), block=True)
        self._live_requests[slot] = request

        self.log(f"Finalized prefill for request {request.id} in slot {slot}")

    def _process_decode_batch(
        self,
        engine,
        decode_state,
        decode_info: DecodeScheduleInfo,
        generate_timestep: int,
    ):
        """Process a batch of decode operations"""
        decode_start = time.perf_counter()

        # Run decode
        decode_state, sampled_tokens = engine.decode(
            graphstate=engine.graphstate,
            graphothers=engine.graphothers,
            state=decode_state,
            cache_metadata=decode_info.page_batch_info,
            rngs=engine.prng_key,
            slot=0,
        )

        decode_duration_ms = (time.perf_counter() - decode_start) * 1000
        self.metrics_recorder.record_decode_op_time(decode_duration_ms)

        sampled_tokens.copy_to_host_async()
        self._detokenize_backlog.put((generate_timestep, sampled_tokens), block=True)

        return decode_state, sampled_tokens

    def _process_prefill_content(
        self,
        request: ActiveRequest,
        processor: ProcessingClassType,
        max_prefill_length: int,
        prefill_lengths: list[int],
        pad_token_id: int,
        do_pad: bool = True,
        right_padding: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, int]:  # Fixed: removed SamplingParams from return type
        """
        Processes the content of a prefill request for the engine.

        This involves tokenizing the input (if it's a string), padding it to
        an appropriate length based on engine capabilities and bucketing.

        Args:
            request: The `ActiveRequest` containing the prompt and settings.
            processor: The tokenizer/processor instance.
            max_prefill_length: Maximum allowed length for prefill.
            prefill_lengths: list of supported prefill bucket lengths.
            pad_token_id: The ID of the padding token.

        Returns:
            A tuple containing: (padded_tokens, padded_valids, true_length)
        """
        content = request.prefill_content
        if isinstance(content, str):
            content = processor(text=content, return_tensors="np", return_attention_mask=True)
            tokens = jnp.array(content["input_ids"])
            valids = jnp.array(content["attention_mask"])
        else:
            tokens = jnp.array(content)
            valids = jnp.ones_like(tokens)

        # Ensure proper shape
        if tokens.ndim == 1:
            tokens = tokens[None, :]  # Add batch dimension
            valids = valids[None, :]

        true_length = tokens.shape[1]  # Fixed: get sequence length, not shape
        if do_pad:
            tokens, valids, true_length = pad_tokens(
                tokens=tokens,
                valids=valids,
                pad_token_id=pad_token_id,
                max_prefill_length=max_prefill_length,
                prefill_lengths=prefill_lengths,
                right_padding=right_padding,
            )
        return tokens, valids, true_length

    def _metrics_monitor_thread_action(self):
        """
        Background thread action for periodically updating and logging metrics.

        This thread updates queue size metrics and logs a summary of aggregated
        metrics at the interval specified by
        `metrics_recorder.metrics_log_interval_sec`. It also attempts to log
        device memory statistics.
        """
        while self.live:
            try:
                self.metrics_recorder.update_queue_size("detokenize_backlog", self._detokenize_backlog.qsize())
                # Fixed: removed reference to non-existent _decode_slots

                active_count = sum(1 for r in self._live_requests.values() if r is not None)
                self.metrics_recorder.set_active_requests_count(active_count)

                if self.metrics_recorder.metrics_log_interval_sec > 0:
                    aggregated_metrics = self.metrics_recorder.get_aggregated_metrics_snapshot(window_size=0)
                    log_summary = {
                        "queues": aggregated_metrics.get("queue_sizes"),
                        "active_reqs": aggregated_metrics.get("active_requests_count"),
                        "submitted_reqs": aggregated_metrics.get("submitted_requests_count"),
                        "completed_reqs": aggregated_metrics.get("completed_requests_count"),
                        "ttft_ms_avg": aggregated_metrics.get("ttft_ms_avg"),
                        "prefill_op_ms_avg": aggregated_metrics.get("prefill_op_ms_avg"),
                        "decode_op_ms_avg": aggregated_metrics.get("decode_op_ms_avg"),
                        "insert_op_ms_avg": aggregated_metrics.get("insert_op_ms_avg"),
                        "prefill_ops_total": aggregated_metrics.get("prefill_ops_count"),
                        "decode_ops_total": aggregated_metrics.get("decode_ops_count"),
                    }
                    log_summary_filtered = {k: v for k, v in log_summary.items() if v is not None}

                    device_stats = self.get_device_memory_stats()
                    if device_stats:
                        log_summary_filtered["device_memory"] = device_stats

                    # self.log(f"Metrics: {log_summary_filtered}")

                sleep_duration = self.metrics_recorder.metrics_log_interval_sec
                if sleep_duration <= 0:
                    sleep_duration = 1.0

                time.sleep(sleep_duration)

            except Exception as e:
                self.log(f"Error in metrics monitor thread: {e} - {traceback.format_exc()}")
                fallback_sleep = 1.0
                try:
                    if self.metrics_recorder.metrics_log_interval_sec > 0:
                        fallback_sleep = self.metrics_recorder.metrics_log_interval_sec
                except AttributeError:
                    pass
                time.sleep(fallback_sleep)

    def replace_graphstate(self, state):
        """
        Replaces the engine's graph state with a new one.

        This is typically used for hot-swapping models or LoRA adapters.

        Args:
            state: The new graph state to be used by the engine.
        """
        self._engine.graphstate = state
        self.log("Replaced engine graph state")

    def start(self):
        """Start the driver and all worker threads"""
        if not self.live:
            self.log("Starting vDriver...")

            # Create threads
            self._inference_thread = SafeThread(
                target=self._unified_inference_thread,  # Fixed: use correct method name
                name="unified-inference-thread",
                daemon=True,
            )

            self._detokenize_thread = SafeThread(
                target=self._detokenize_action_thread,
                name="detokenize-thread",
                daemon=True,
            )

            self._all_threads = [self._inference_thread, self._detokenize_thread]

            # Set live flag
            self.live = True

            # Start threads
            for thread in self._all_threads:
                if not thread.is_alive():
                    self.log(f"Starting thread: {thread.name}")
                    thread.start()

            # Start metrics thread
            if self._metrics_thread is None or not self._metrics_thread.is_alive():
                self._metrics_thread = SafeThread(
                    target=self._metrics_monitor_thread_action,
                    name="metrics-monitor-thread",
                    daemon=True,
                )
                self.log("Starting metrics monitor thread")
                self._metrics_thread.start()
                self._all_threads.append(self._metrics_thread)

            self.log("vDriver started successfully")

    def stop(self):
        """Stop the driver and all worker threads gracefully"""
        if self.live:
            self.log("Stopping vDriver...")
            self.live = False

            # Signal threads to stop
            try:
                self._detokenize_backlog.put_nowait(None)
            except queue.Full:
                self.log("Detokenize queue full while sending stop signal")

            # Join threads
            for thread in self._all_threads:
                if thread.is_alive():
                    self.log(f"Joining thread: {thread.name}")
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        self.log(f"Warning: Thread {thread.name} did not stop cleanly")

            # Clean up any remaining requests
            self._cleanup_active_requests()

            self.log("vDriver stopped")

    def _cleanup_active_requests(self):
        """Clean up any active requests by closing their channels"""
        for slot, request in self._live_requests.items():
            if request and request.return_channel:
                self.log(f"Closing return channel for request in slot {slot}")
                request.return_channel.close()

        # Also check scheduler for pending requests
        if hasattr(self.scheduler, "active_requests"):
            for request_id, request in self.scheduler.active_requests.items():
                if request and request.return_channel:
                    self.log(f"Closing return channel for pending request {request_id}")
                    request.return_channel.close()

    def resume(self):
        """
        Resumes the vDriver if it was previously paused.

        Re-initializes the scheduler queues and starts all worker threads.
        The metrics recorder continues accumulating from its previous state.
        """
        if self._pause:
            self.log("Resuming vDriver...")
            self.start()
            self._pause = False
            self.log("vDriver resumed.")

    def pause(self):
        """
        Pauses the vDriver.

        Stops all worker threads and sets the `_pause` flag to True.
        Requests submitted while paused will be queued but not processed
        until `resume()` is called.
        """
        if not self._pause:
            self.log("Pausing vDriver...")
            self.stop()
            self._pause = True
            self.log("vDriver paused.")

    def submit_request(self, request: ActiveRequest, priority: RequestPriority = RequestPriority.NORMAL):
        """
        Submits a new request for processing by the vDriver.

        Args:
            request: The request to submit. Must be an instance of `ActiveRequest`.
            priority: Priority level for the request.

        Raises:
            TypeError: If the submitted request is not an `ActiveRequest`.
        """
        if not isinstance(request, ActiveRequest):
            raise TypeError("Request must be of type ActiveRequest")
        if not self.live and not self._pause:
            self.log("Warning: Driver is not live and not paused. Request may not be processed.")

        self.log(f"Submitting new request {request.id} with priority {priority.name}")
        self.metrics_recorder.increment_submitted_requests()
        self._request_counter += 1
        self.scheduler.add_request(request=request, priority=priority)

    def get_metrics(self, aggregated: bool = True, window_size: int = 100) -> dict:
        """
        Returns a snapshot of the current metrics from the `MetricsRecorder`.

        Args:
            aggregated: If True, returns aggregated metrics (avg, p50, p99 etc.)
                        calculated by `MetricsRecorder`. Defaults to True.
            window_size: The number of recent samples to use for aggregation if
                         `aggregated` is True. Set to 0 to use all samples.
                         Defaults to 100.

        Returns:
            dict: A dictionary containing either raw or aggregated metrics.
        """
        if aggregated:
            metrics = self.metrics_recorder.get_aggregated_metrics_snapshot(window_size)
        else:
            metrics = self.metrics_recorder.get_all_metrics()

        # Add scheduler stats if available
        if hasattr(self.scheduler, "get_stats"):
            metrics["scheduler"] = self.scheduler.get_stats()

        return metrics

    def get_device_memory_stats(self) -> list[dict] | None:
        """
        Attempts to retrieve memory statistics for the primary local JAX device.

        This relies on the `memory_stats()` method of the JAX device object,
        which may not be available on all platforms or JAX versions.

        Returns:
            Optional[list[dict]]: A list of dictionaries with memory statistics (e.g.,
                'bytes_used', 'bytes_limit') if available, otherwise None.
        """
        try:
            local_devs = jax.local_devices()
            if not local_devs:
                self.log("No local JAX devices found.")
                return None
            infos = []
            for device in local_devs:
                if hasattr(device, "memory_stats"):
                    stats = device.memory_stats()
                    bytes_available = stats.get("bytes_limit", 0) - stats.get("bytes_used", 0)
                    infos.append(
                        {
                            "device_id": device.id,
                            "platform": device.platform,
                            "device_kind": device.device_kind,
                            "bytes_limit": stats.get("bytes_limit"),
                            "bytes_used": stats.get("bytes_used"),
                            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
                            "bytes_available": bytes_available,
                        }
                    )
                else:
                    self.log(f"Device {device.id} ({device.device_kind}) object does not have memory_stats method.")
            return infos if infos else None
        except Exception as e:
            self.log(f"Could not retrieve JAX device memory stats: {e}")
            return None
