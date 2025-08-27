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

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

from ...sampling_params import JitableSamplingParams
from ..utils import ActiveRequest, MetricsRecorder, ReturnSample, SafeThread, pad_tokens, process_result_tokens
from .engine import vEngine
from .scheduler import Scheduler, SchedulerAction

if tp.TYPE_CHECKING:
    from easydel.infra.utils import ProcessingClassType

logger = get_logger("vSurge-vDriver")

DEFAULT_PREFILL_BACKLOG_MAXSIZE = 0
DEFAULT_TRANSFER_BACKLOG_MAXSIZE = None
DEFAULT_DECODE_BACKLOG_MAXSIZE = None
DEFAULT_METRICS_LOG_INTERVAL_SEC = 120.0
DEFAULT_DETOKENIZING_BLOCKS = 8
DEFAULT_SLOT_CLEAR_STEPS = 512
MIN_SLEEP_DURATION = 0.001  # Minimum sleep time to prevent busy waiting
MAX_THREAD_JOIN_TIMEOUT = 3.0  # Maximum timeout for thread joining


class vDriver:
    """
    Drives the vEngine for prefill and decode operations, managing request flow.
    Optimized to reduce Python overhead through better thread management,
    memory efficiency, and JAX-specific optimizations.
    """

    _engine: vEngine
    scheduler: Scheduler
    _process_thread: SafeThread
    _detokenize_thread: SafeThread
    _metrics_thread: SafeThread | None
    _all_threads: list[SafeThread]

    def __init__(
        self,
        engine: vEngine,
        interleaved_mode: bool = False,
        slot_clear_steps: int = DEFAULT_SLOT_CLEAR_STEPS,
        verbose: bool = True,
        prefill_backlog_maxsize: int = DEFAULT_PREFILL_BACKLOG_MAXSIZE,
        transfer_backlog_maxsize: int | None = DEFAULT_TRANSFER_BACKLOG_MAXSIZE,
        decode_backlog_maxsize: int | None = DEFAULT_DECODE_BACKLOG_MAXSIZE,
        metrics_log_interval_sec: float = DEFAULT_METRICS_LOG_INTERVAL_SEC,
    ):
        """Initializes the vDriver with optimized defaults."""
        self._pause = False
        self._engine = engine
        self._interleaved_mode = interleaved_mode
        self._slot_clear_steps = slot_clear_steps
        self.prefill_backlog_maxsize = prefill_backlog_maxsize
        self.transfer_backlog_maxsize = transfer_backlog_maxsize
        self.decode_backlog_maxsize = decode_backlog_maxsize
        self.metrics_recorder = MetricsRecorder(metrics_log_interval_sec)
        self._request_counter = 0
        self.scheduler = Scheduler(self._engine)

        # Only create the process thread - no detokenize thread
        self._process_thread = SafeThread(target=self._process_action_thread, name="process-thread", daemon=True)
        self.log = logger.info if verbose else logger.debug
        self.live = False
        self._metrics_thread = None

    @property
    def engine(self) -> vEngine:
        return self._engine

    @property
    def driver_name(self) -> str:
        """Returns a standardized name for the driver and its model."""
        return self._get_model_name(self._engine.model)

    def place_request_on_prefill_queue(self, request: ActiveRequest):
        """Legacy method: Places a request onto the legacy prefill backlog."""
        self.scheduler.add_request(request)

    def submit_request(self, request: tp.Any):
        """Submits a new request for processing by the vDriver."""
        if not isinstance(request, ActiveRequest):
            raise TypeError("Request must be of type ActiveRequest")
        if not self.live and not self._pause:
            self.log("[Main] Warning: Driver is not live and not paused. Request may not be processed.")
        self.log("[Main] Submitting a new request.")
        self.metrics_recorder.increment_submitted_requests()
        self.scheduler.add_request(request)

    @property
    def processor(self) -> ProcessingClassType:
        """Returns the tokenizer/processor associated with the engine."""
        return self._engine.processor

    def _calculate_model_size(self, graphstate) -> str:
        """Calculates the model size in billions of parameters."""
        try:
            num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
            size_in_billions = num_params / 1e9
            return f"{size_in_billions:.2f}"
        except Exception:
            return "unknown"

    def _get_model_name(self, model) -> str:
        """Generates a model name string including type and size."""
        model_type = self._get_model_type(model)
        model_size = self._calculate_model_size(model.graphstate)
        return f"{model_type}-{model_size}b"

    def _get_model_type(self, model) -> str:
        """Extracts the model type from the model's configuration."""
        return getattr(model.config, "model_type", "unknown").lower()

    def compile(self):
        """Compiles the prefill, insert, and decode functions of the engine."""
        engine = self._engine
        try:
            decode_state = engine.init_decode_state()
            # Prefetch all prefill lengths to compile all necessary functions
            for length in engine.prefill_lengths:
                padded_tokens = padded_valids = jnp.ones((1, length), "i4")
                self.log(f"[Compile] Compiling prefill/insert length={length}")
                state_new, _ = engine.prefill(
                    graphstate=engine.graphstate,
                    graphothers=engine.graphothers,
                    tokens=padded_tokens,
                    valids=padded_valids,
                    true_length=0,
                    cache=None,
                    cache_metadata=None,
                    sampling_params=JitableSamplingParams.init_empty(1).view_1d(),
                    rngs=engine.prng_key,
                    slot=0,
                )
                decode_state = engine.insert(state_new, decode_state, 0)

            self.log("[Compile] Compiling decode")
            decode_state = engine.free_state_resources([0], decode_state)
            decode_state = engine.decode(
                graphstate=engine.graphstate,
                graphothers=engine.graphothers,
                state=decode_state,
                cache_metadata=None,
                rngs=engine.prng_key,
                slot=0,
            )
            engine.free_resource(0)
            del decode_state
        except Exception:
            traceback.print_exc()
            self.stop()
            exit(1)

    def get_total_concurrent_requests(self) -> int:
        """Returns the total number of concurrent decode requests the engine can handle."""
        return self._engine.total_max_concurrent_decodes

    def _jax_transfer_prefill_result(self, new_request: ActiveRequest):
        """Transfers prefill results (KV cache) to the engine's device."""
        start_time = time.perf_counter()
        dst_sharding = self._engine.get_prefix_destination_sharding()
        new_request.prefill_result = jax.device_put(new_request.prefill_result, dst_sharding)
        jax.block_until_ready(new_request.prefill_result)
        duration_ms = (time.perf_counter() - start_time) * 1000
        self.metrics_recorder.record_transfer_op_time(duration_ms)

    def _process_decode_slots(self, decode_slots: list, engine: vEngine, decode_state, generate_timestep: int):
        """Process decode - just generate and queue for detokenization."""
        time_before_decode_call = time.perf_counter()

        try:
            decode_state, sampled_tokens = engine.decode(
                graphstate=engine.graphstate,
                graphothers=engine.graphothers,
                cache_metadata=None,
                state=decode_state,
                rngs=engine.prng_key,
                slot=0,
            )
        except Exception as e:
            self.log(f"[Process] ERROR during decode step {generate_timestep}: {e}", exc_info=True)
            time.sleep(0.01)
            return generate_timestep, decode_state

        decode_op_duration_ms = (time.perf_counter() - time_before_decode_call) * 1000
        self.metrics_recorder.record_decode_op_time(decode_op_duration_ms)
        sampled_tokens.copy_to_host_async()

        try:
            self._detokenize_queue.put((generate_timestep, sampled_tokens), block=True)
        except queue.Full:
            self.log("[Process] Detokenize queue full - this shouldn't happen")

        generate_timestep += 1

        total_decode_cycle_ms = (time.perf_counter() - time_before_decode_call) * 1000
        self.log(
            f"[Process] Decode step {generate_timestep} completed - "
            f"Active requests: {len(decode_slots)}, "
            f"Decode op time: {decode_op_duration_ms:.2f}ms, "
            f"Detokenize op time: {(total_decode_cycle_ms - decode_op_duration_ms):.2f}ms, "
            f"Total cycle time: {total_decode_cycle_ms:.2f}ms"
        )

        return generate_timestep, decode_state

    def _insert_request_into_slot(
        self,
        request: ActiveRequest,
        slot: int,
        engine: vEngine,
        decode_state,
    ):
        """Insert request and register it for detokenization."""
        self.log(f"[Process] Inserting prefilled request into decode slot {slot}.")
        insert_start_time = time.perf_counter()

        try:
            decode_state = engine.insert(
                prefix=request.prefill_result,
                decode_state=decode_state,
                slot=slot,
            )
        except Exception as e:
            self.log(f"[Process] ERROR during insert into slot {slot}: {e}", exc_info=True)
            del request.prefill_result
            return decode_state

        insert_duration_ms = (time.perf_counter() - insert_start_time) * 1000
        self.metrics_recorder.record_insert_op_time(insert_duration_ms)
        del request.prefill_result

        try:
            self._detokenize_queue.put((slot, request), block=True)
        except queue.Full:
            self.log("[Process] Detokenize queue full when inserting request")

        self.scheduler.insert_prefill_result(request, slot)
        self.log(
            f"[Process] Request successfully inserted into slot {slot}. Insert duration: {insert_duration_ms:.2f}ms"
        )
        return decode_state

    def _detokenize_thread_action(self):
        """Dedicated detokenization thread."""
        engine = self._engine
        processor = engine.processor

        live_requests = {i: None for i in range(engine.max_concurrent_decodes)}

        self.log("[Detokenize] Detokenization thread started.")

        while self.live:
            try:
                data = self._detokenize_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if data is None:
                break

            if len(data) == 3 and isinstance(data[1], ActiveRequest):
                first_token, request, _ = data
                self._process_first_token(first_token, request, processor)
                continue

            if isinstance(data[1], ActiveRequest):
                slot, request = data
                live_requests[slot] = request
                self.log(f"[Detokenize] Registered request for slot {slot}")
                continue

            if len(data) == 2:
                generate_timestep, result_tokens = data
                result_tokens_np = result_tokens.convert_to_numpy()

                # Process all live slots
                for slot, request in live_requests.items():
                    if request is None:
                        continue

                    try:
                        self._process_slot_tokens(slot, request, result_tokens_np, processor, engine)

                        # Check if request is complete
                        if request.complete.all():
                            if request.return_channel:
                                request.return_channel.close()
                            self.log(f"[Detokenize] Request in slot {slot} completed.")
                            self.metrics_recorder.increment_completed_requests()
                            self.scheduler.free_slot(slot)
                            engine.free_resource(slot)
                            live_requests[slot] = None

                    except Exception as e:
                        self.log(f"[Detokenize] Error processing slot {slot}: {e}")

        self.log("[Detokenize] Detokenization thread stopped.")

    def _process_slot_tokens(self, slot: int, request: ActiveRequest, result_tokens_np, processor, engine):
        """Process tokens for a single slot."""
        if request.decode_start_time is None:
            request.decode_start_time = time.perf_counter()

        # Use the existing process_result_tokens function
        results_base, complete, num_valid_tokens_list = process_result_tokens(
            processor=processor,
            slot=slot,
            slot_max_length=request.sampling_params.max_tokens,
            result_tokens=result_tokens_np,
            eos_token_id=engine.eos_token_ids,
            is_client_side_tokenization=request.is_client_side_tokenization,
            complete=request.complete,
            ignore_eos=request.sampling_params.ignore_eos,
        )

        request.complete = complete
        elapsed_time = time.perf_counter() - request.decode_start_time

        for res_base, num_valid in zip(results_base, num_valid_tokens_list, strict=False):
            if len(res_base.text) > 0:
                request.add_text_fragments(res_base.text)

            # Check stop conditions
            if request.sampling_params.stop is not None:
                for stop_sign in request.sampling_params.stop:
                    for idx, accum in enumerate(request.accumulated_text):
                        if stop_sign in accum:
                            request.complete[idx] = True

            request.total_generated_tokens += num_valid
            tps = request.total_generated_tokens / elapsed_time if elapsed_time > 1e-6 else 0.0

            if request.return_channel:
                result = ReturnSample(
                    text=res_base.text,
                    token_ids=res_base.token_ids,
                    time_spent_computing=elapsed_time,
                    accumulated_text=request.accumulated_text,
                    tokens_per_second=tps,
                    num_generated_tokens=request.total_generated_tokens,
                )
                request.enqueue_samples([result])

    def _process_first_token(self, first_token, request: ActiveRequest, processor):
        """Process first token from prefill."""
        first_token_start = time.perf_counter()
        first_token_np = first_token.convert_to_numpy()

        if not hasattr(request, "complete") or request.complete is None:
            request.complete = np.zeros((self._engine.samples_per_slot,), dtype=np.bool_)

        results_base, complete, num_valid_tokens_list = process_result_tokens(
            processor=processor,
            slot=0,
            slot_max_length=request.sampling_params.max_tokens,
            result_tokens=first_token_np,
            eos_token_id=self._engine.eos_token_ids,
            is_client_side_tokenization=request.is_client_side_tokenization,
            complete=request.complete,
            ignore_eos=request.sampling_params.ignore_eos,
        )

        request.complete = complete

        for res_base, num_valid in zip(results_base, num_valid_tokens_list, strict=False):
            if isinstance(res_base.text, list):
                request.add_text_fragments(res_base.text)
            else:
                text_list = [str(res_base.text)] if res_base.text is not None else [""]
                request.add_text_fragments(text_list)

            request.total_generated_tokens += num_valid

            if request.return_channel:
                result = ReturnSample(
                    text=res_base.text,
                    token_ids=res_base.token_ids,
                    time_spent_computing=0.0,
                    accumulated_text=request.accumulated_text,
                    tokens_per_second=0.0,
                    num_generated_tokens=request.total_generated_tokens,
                )
                request.enqueue_samples([result])

        first_token_duration = (time.perf_counter() - first_token_start) * 1000
        self.metrics_recorder.record_ttft(first_token_duration)
        self.log(f"[Detokenize] TTFT: {first_token_duration:.2f}ms for request.")

    def _process_prefill_content(
        self,
        request: ActiveRequest,
        processor: ProcessingClassType,
        max_prefill_length: int,
        prefill_lengths: list[int],
        pad_token_id: int,
        do_pad: bool = True,
        right_padding: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, int]:
        """Processes the content of a prefill request for the engine."""
        content = request.prefill_content
        if isinstance(content, str):
            # Use a more efficient tokenization approach
            content = processor(text=content, return_tensors="np", return_attention_mask=True)
            tokens = jnp.array(content["input_ids"])
            valids = jnp.array(content["attention_mask"])
        else:
            tokens, valids = content

        true_length = len(tokens)
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

    def _process_action_thread(self):
        """
        Background thread action for performing prefill, transferring KV cache,
        inserting into decode slots, and performing decode steps.
        """
        engine = self._engine
        processor = engine.processor
        generate_timestep = 0
        decode_state = engine.init_decode_state()
        self.log("[Process] Processing thread started.")

        last_slot_clear_step = 0

        while self.live:
            action: SchedulerAction = self.scheduler.schedule()
            for request in action.prefill_requests:
                decode_state = self._process_single_prefill_request(request, engine, processor, decode_state)

            if action.decode_slots:
                generate_timestep, decode_state = self._process_decode_slots(
                    action.decode_slots,
                    engine,
                    decode_state,
                    generate_timestep,
                )
            if self._slot_clear_steps and (generate_timestep - last_slot_clear_step) >= self._slot_clear_steps:
                decode_state = self._perform_slot_cleanup(engine, decode_state, generate_timestep)
                last_slot_clear_step = generate_timestep

        self.log("[Process] Processing thread stopped.")

    def _process_single_prefill_request(
        self,
        request: ActiveRequest,
        engine: vEngine,
        processor: ProcessingClassType,
        decode_state,
    ):
        """Process a single prefill request with inline first token processing."""
        self.log("[Process] Starting prefill for request.")
        padded_tokens, padded_valids, true_length = self._process_prefill_content(
            request,
            processor,
            engine.max_prefill_length,
            engine.prefill_lengths,
            engine.pad_token_id,
        )
        num_tokens = padded_valids.shape[-1]
        self.log(f"[Process] Prefill processing content, tokens: {num_tokens}")

        # Perform prefill
        prefill_start_time = time.perf_counter()
        try:
            prefill_result, first_token = engine.prefill(
                graphstate=engine.graphstate,
                graphothers=engine.graphothers,
                tokens=padded_tokens,
                cache=None,
                cache_metadata=None,
                valids=padded_valids,
                true_length=true_length,
                sampling_params=request.sampling_params.make_jitable().view_1d(),
                rngs=engine.prng_key,
                slot=0,
            )
        except Exception as e:
            self.log(f"[Process] ERROR during prefill: {e}", exc_info=True)
            return decode_state

        prefill_duration_ms = (time.perf_counter() - prefill_start_time) * 1000
        self.metrics_recorder.record_prefill_op_time(prefill_duration_ms)
        request.prefill_result = prefill_result

        # Process first token inline immediately
        self._process_first_token_inline(first_token, request, engine, processor)

        self.log(f"[Process] Prefill completed, tokens: {num_tokens}, duration: {prefill_duration_ms:.2f}ms")

        # Handle non-interleaved mode
        if not self._interleaved_mode:
            self.log("[Process] Transferring KV cache (non-interleaved).")
            transfer_start_time = time.perf_counter()
            try:
                self._transfer_prefill_result(request)
            except Exception as e:
                self.log(f"[Process] ERROR during transfer: {e}", exc_info=True)
                del request.prefill_result
                return decode_state
            transfer_duration_ms = (time.perf_counter() - transfer_start_time) * 1000
            self.log(f"[Process] Transfer completed, duration: {transfer_duration_ms:.2f}ms")

        # Try to insert into a free slot
        if self.scheduler.has_free_slot():
            free_slot = self._find_free_slot()
            if free_slot is not None:
                decode_state = self._insert_request_into_slot(request, free_slot, engine, decode_state)
            else:
                self.log("[Process] ERROR: Scheduler scheduled a prefill but no free slot found!")
                del request.prefill_result
        return decode_state

    def _find_free_slot(self) -> int | None:
        """Find a free slot in the scheduler."""
        for potential_slot in range(self._engine.max_concurrent_decodes):
            if potential_slot not in self.scheduler._live_requests:
                if potential_slot in self.scheduler._free_slots:
                    return potential_slot
        return None

    def _process_first_token_inline(self, first_token, request: ActiveRequest, engine: vEngine, processor):
        """Process first token inline using the new text fragments approach."""
        first_token_start = time.perf_counter()
        first_token_np = first_token.convert_to_numpy()
        if not hasattr(request, "complete") or request.complete is None:
            request.complete = np.zeros((engine.samples_per_slot,), dtype=np.bool_)
        results_base, complete, num_valid_tokens_list = process_result_tokens(
            processor=processor,
            slot=0,
            slot_max_length=request.sampling_params.max_tokens,
            result_tokens=first_token_np,
            eos_token_id=engine.eos_token_ids,
            is_client_side_tokenization=request.is_client_side_tokenization,
            complete=request.complete,
            ignore_eos=request.sampling_params.ignore_eos,
        )
        request.complete = complete
        final_results = []
        for res_base, num_valid in zip(results_base, num_valid_tokens_list, strict=False):
            # Initialize text fragments with first token
            if isinstance(res_base.text, list):
                request.add_text_fragments(res_base.text)
            else:
                text_list = [str(res_base.text)] if res_base.text is not None else [""]
                request.add_text_fragments(text_list)

            request.total_generated_tokens += num_valid
            final_results.append(
                ReturnSample(
                    text=res_base.text,
                    token_ids=res_base.token_ids,
                    time_spent_computing=0.0,
                    accumulated_text=request.accumulated_text,  # Uses the property
                    tokens_per_second=0.0,
                    num_generated_tokens=request.total_generated_tokens,
                )
            )
        if request.return_channel:
            request.enqueue_samples(final_results)
        first_token_duration = (time.perf_counter() - first_token_start) * 1000
        self.metrics_recorder.record_ttft(first_token_duration)
        self.log(f"[Process] TTFT: {first_token_duration:.2f}ms for request.")

    def _perform_slot_cleanup(self, engine: vEngine, decode_state, generate_timestep: int):
        """Perform periodic cleanup of unused slot resources."""
        self.log(f"[Process] Decode step {generate_timestep}: Performing periodic slot resource cleanup.")
        try:
            free_slots_list = list(self.scheduler._free_slots)
            if free_slots_list:
                decode_state = engine.free_state_resources(free_slots_list, decode_state)
        except Exception as e:
            self.log(f"[Process] WARNING during slot cleanup at step {generate_timestep}: {e}", exc_info=True)
        self.log(f"[Process] Slot cleanup completed at step {generate_timestep}.")
        return decode_state

    def _metrics_monitor_thread_action(self):
        """Background thread action for periodically updating and logging metrics."""
        while self.live:
            try:
                self.metrics_recorder.update_queue_size("free_decode_slots", self.scheduler.get_free_slot_count())
                self.metrics_recorder.set_active_requests_count(self.scheduler.get_active_request_count())

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

                    self.log(f"[Metrics] {log_summary_filtered}")

                sleep_duration = self.metrics_recorder.metrics_log_interval_sec
                if sleep_duration <= 0:
                    sleep_duration = 1.0
                time.sleep(sleep_duration)
            except Exception as e:
                self.log(f"[Metrics] Error in metrics monitor thread: {e} - {traceback.format_exc()}")
                fallback_sleep = 1.0
                try:
                    if self.metrics_recorder.metrics_log_interval_sec > 0:
                        fallback_sleep = self.metrics_recorder.metrics_log_interval_sec
                except AttributeError:
                    pass
                time.sleep(fallback_sleep)

    def replace_graphstate(self, state):
        """Replaces the engine's graph state with a new one."""
        self._engine.graphstate = state

    def start(self):
        """Starts the vDriver with process and detokenize threads."""
        if not self.live:
            self.log("[Main] Starting vDriver...")
            self._detokenize_queue = queue.Queue(maxsize=8)

            self._detokenize_thread = SafeThread(
                target=self._detokenize_thread_action,
                name="detokenize-thread",
                daemon=True,
            )

            self._all_threads = [self._process_thread, self._detokenize_thread]
            self.live = True

            for t in self._all_threads:
                if not t.is_alive():
                    self.log(f"[Main] Starting thread: {t.name}")
                    t.start()

            if self._metrics_thread is None or not self._metrics_thread.is_alive():
                self._metrics_thread = SafeThread(
                    target=self._metrics_monitor_thread_action,
                    name="metrics-monitor-thread",
                    daemon=True,
                )
                self.log("[Main] Starting metrics monitor thread.")
                self._metrics_thread.start()
                self._all_threads.append(self._metrics_thread)

            self.log("[Main] vDriver started.")

    def stop(self):
        """Stop all threads and clean up queues."""
        if self.live:
            self.log("[Main] Stopping vDriver...")
            self.live = False

            # Send sentinel to detokenize queue
            try:
                self._detokenize_queue.put_nowait(None)
            except queue.Full:
                pass

            # Join all threads
            current_threads_to_join = list(self._all_threads)
            for t in current_threads_to_join:
                if t.is_alive():
                    self.log(f"[Main] Joining thread: {t.name}")
                    t.join(timeout=MAX_THREAD_JOIN_TIMEOUT)

            self.log("[Main] vDriver stopped.")

    def resume(self):
        """Resumes the vDriver if it was previously paused."""
        if self._pause:
            self.log("[Main] Resuming vDriver...")
            self.start()
            self._pause = False
            self.log("[Main] vDriver resumed.")

    def pause(self):
        """Pauses the vDriver."""
        if not self._pause:
            self.log("[Main] Pausing vDriver...")
            self.stop()
            self._pause = True
            self.log("[Main] vDriver paused.")

    def _transfer_prefill_result(self, new_request: ActiveRequest):
        """Helper method to invoke the JAX-specific KV cache transfer."""
        self._jax_transfer_prefill_result(new_request)

    def get_metrics(self, aggregated: bool = True, window_size: int = 100) -> dict:
        """Returns a snapshot of the current metrics from the `MetricsRecorder`."""
        if aggregated:
            return self.metrics_recorder.get_aggregated_metrics_snapshot(window_size)
        else:
            return self.metrics_recorder.get_all_metrics()

    def get_device_memory_stats(self) -> list[dict] | None:
        """Attempts to retrieve memory statistics for the primary local JAX device."""
        try:
            local_devs = jax.local_devices()
            if not local_devs:
                self.log("[Main] No local JAX devices found.")
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
                            "bytes_available": stats.get("bytes_available", bytes_available),
                        }
                    )
                else:
                    self.log(
                        f"[Main] Device {device.id} ({device.device_kind}) object does not have memory_stats method."
                    )

            if len(infos) == 0:
                return None
            return infos
        except Exception as e:
            self.log(f"[Main] Could not retrieve JAX device memory stats: {e}")
            return None
