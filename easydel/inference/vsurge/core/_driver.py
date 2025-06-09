# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
import threading
import time
import traceback
import typing as tp

import jax
import numpy as np
from jax import numpy as jnp

from easydel.inference.utilities import SamplingParams
from easydel.utils.helpers import get_logger

from ..utils import (
    ActiveRequest,
    ResultTokens,
    ReturnSample,
    SafeThread,
    pad_tokens,
    process_result_tokens,
)
from ._engine import vEngine

if tp.TYPE_CHECKING:
    from easydel.infra.utils import ProcessingClassType
else:
    ProcessingClassType = tp.Any

logger = get_logger("vSurge-vDriver")


class MetricsRecorder:
    """
    Records and provides access to various operational metrics.

    This class is responsible for collecting time-series data for various
    durations, counts, and queue sizes within the vDriver system. It provides
    methods to update these metrics and retrieve them in raw or aggregated forms.

    Attributes:
        metrics (dict): A dictionary holding all recorded metrics.
        metrics_log_interval_sec (float): Interval for logging metrics by a monitor.
        _lock (threading.Lock): A lock to ensure thread-safe updates to metrics.
        _max_list_len (int): Maximum length for lists storing time-series data
                             to prevent unbounded memory growth.
    """

    def __init__(self, metrics_log_interval_sec: float = 10.0):
        """
        Initializes the MetricsRecorder.

        Args:
            metrics_log_interval_sec (float): The interval in seconds at which
                a monitoring thread might log these metrics. Defaults to 10.0.
        """
        self.metrics = {
            "queue_sizes": {},
            "active_requests_count": 0,
            "ttft_ms": [],
            "prefill_op_ms": [],
            "decode_op_ms": [],
            "insert_op_ms": [],
            "transfer_op_ms": [],
            "operation_lock_wait_ms": [],
            "prefill_ops_count": 0,
            "decode_ops_count": 0,
            "insert_ops_count": 0,
            "completed_requests_count": 0,
            "submitted_requests_count": 0,
        }
        self._lock = threading.Lock()
        self.metrics_log_interval_sec = metrics_log_interval_sec
        self._max_list_len = 1000

    def _append_to_list(self, key: str, value: float):
        """
        Appends a value to a list metric, ensuring it doesn't exceed max length.

        Args:
            key (str): The key of the list metric in `self.metrics`.
            value (float): The value to append.
        """
        lst = self.metrics.get(key, [])
        lst.append(value)
        self.metrics[key] = lst[-self._max_list_len :]

    def update_queue_size(self, queue_name: str, size: int):
        """
        Updates the recorded size for a specific queue.

        Args:
            queue_name (str): The name of the queue.
            size (int): The current size of the queue.
        """
        with self._lock:
            self.metrics["queue_sizes"][queue_name] = size

    def set_active_requests_count(self, count: int):
        """
        Sets the current count of active requests.

        Args:
            count (int): The number of currently active requests.
        """
        with self._lock:
            self.metrics["active_requests_count"] = count

    def record_ttft(self, ttft_ms: float):
        """
        Records a Time To First Token (TTFT) duration.

        Args:
            ttft_ms (float): The TTFT duration in milliseconds.
        """
        with self._lock:
            self._append_to_list("ttft_ms", ttft_ms)

    def record_prefill_op_time(self, duration_ms: float):
        """
        Records the duration of a prefill operation and increments its count.

        Args:
            duration_ms (float): The prefill operation duration in milliseconds.
        """
        with self._lock:
            self._append_to_list("prefill_op_ms", duration_ms)
            self.metrics["prefill_ops_count"] += 1

    def record_decode_op_time(self, duration_ms: float):
        """
        Records the duration of a decode operation and increments its count.

        Args:
            duration_ms (float): The decode operation duration in milliseconds.
        """
        with self._lock:
            self._append_to_list("decode_op_ms", duration_ms)
            self.metrics["decode_ops_count"] += 1

    def record_insert_op_time(self, duration_ms: float):
        """
        Records the duration of an insert operation and increments its count.

        Args:
            duration_ms (float): The insert operation duration in milliseconds.
        """
        with self._lock:
            self._append_to_list("insert_op_ms", duration_ms)
            self.metrics["insert_ops_count"] += 1

    def record_transfer_op_time(self, duration_ms: float):
        """
        Records the duration of a transfer operation.

        Args:
            duration_ms (float): The transfer operation duration in milliseconds.
        """
        with self._lock:
            self._append_to_list("transfer_op_ms", duration_ms)

    def record_operation_lock_wait_time(self, duration_ms: float):
        """
        Records the time spent waiting for an operation lock.

        Args:
            duration_ms (float): The lock wait duration in milliseconds.
        """
        with self._lock:
            self._append_to_list("operation_lock_wait_ms", duration_ms)

    def increment_completed_requests(self):
        """Increments the count of completed requests."""
        with self._lock:
            self.metrics["completed_requests_count"] += 1

    def increment_submitted_requests(self):
        """Increments the count of submitted requests."""
        with self._lock:
            self.metrics["submitted_requests_count"] += 1

    def get_all_metrics(self) -> dict:
        """
        Returns a deep copy of all currently recorded metrics.

        Returns:
            dict: A copy of the metrics dictionary.
        """
        with self._lock:
            copied_metrics = {}
            for k, v in self.metrics.items():
                if isinstance(v, list):
                    copied_metrics[k] = list(v)
                elif isinstance(v, dict):
                    copied_metrics[k] = dict(v)
                else:
                    copied_metrics[k] = v
            return copied_metrics

    def get_aggregated_metrics_snapshot(self, window_size=100) -> dict:
        """
        Returns a snapshot of aggregated metrics.

        For list-based metrics (e.g., durations), it calculates average,
        percentiles (p50, p90, p99), min, and max over a specified window
        of recent samples.

        Args:
            window_size (int): The number of recent samples to use for
                aggregation. If 0, all samples are used. Defaults to 100.

        Returns:
            dict: A dictionary of aggregated metrics.
        """
        snapshot = self.get_all_metrics()
        aggregated = {}
        for key, value in snapshot.items():
            if isinstance(value, list) and value:
                sample = value[-window_size:] if window_size > 0 else value
                if sample:
                    aggregated[f"{key}_avg"] = round(np.mean(sample), 2)
                    aggregated[f"{key}_p50"] = round(np.percentile(sample, 50), 2)
                    aggregated[f"{key}_p90"] = round(np.percentile(sample, 90), 2)
                    aggregated[f"{key}_p99"] = round(np.percentile(sample, 99), 2)
                    aggregated[f"{key}_min"] = round(np.min(sample), 2)
                    aggregated[f"{key}_max"] = round(np.max(sample), 2)
                    aggregated[f"{key}_count_total"] = snapshot.get(f"{key.split('_ms')[0]}_ops_count", len(value))
                    aggregated[f"{key}_count_window"] = len(sample)
            elif isinstance(value, dict):
                aggregated[key] = dict(value)
            elif isinstance(value, int | float):
                aggregated[key] = value
        return aggregated


class vDriver:
    """
    Drives the vEngine for prefill and decode operations, managing request flow.

    The `vDriver` orchestrates the entire inference pipeline, including request
    submission, prefilling, KV cache transfer, decoding, and detokenization.
    It uses a series of background threads and queues to manage these stages
    concurrently. It also incorporates a `MetricsRecorder` to track operational
    statistics.

    Attributes:
        _engine (vEngine): The underlying engine performing model computations.
        _prefill_backlog (queue.Queue): Queue for incoming prefill requests.
        _transfer_backlog (queue.Queue): Queue for requests awaiting KV cache transfer.
        _decode_backlog (queue.Queue): Queue for requests ready for decoding.
        _detokenize_backlog (queue.Queue): Queue for results awaiting detokenization.
        _decode_slots (queue.Queue): Queue managing available decode slots in the engine.
        _live_requests (dict): Tracks requests currently active in decode slots.
        _interleaved_mode (bool): If True, prioritizes new requests for lower latency.
        _slot_clear_steps (int): Interval for clearing unused resources in decode state.
        _detokenizing_blocks (int): Max size for the detokenize backlog.
        _use_operation_lock (bool): If True, serializes device operations (prefill,
                                    insert, decode) to prevent OOMs.
        metrics_recorder (MetricsRecorder): Instance for recording metrics.
        log (function): Logger instance for outputting messages.
        live (bool): Flag indicating if the driver's worker threads are active.
        _pause (bool): Flag indicating if the driver is paused.
        _operation_lock (threading.Lock | None): Lock for serializing device ops.
        _all_threads (list[SafeThread]): List of all managed background threads.
        _metrics_thread (SafeThread | None): Thread for monitoring and logging metrics.
    """

    _engine: vEngine
    _prefill_backlog: queue.Queue[ActiveRequest | None]
    _transfer_backlog: queue.Queue[ActiveRequest]
    _decode_backlog: queue.Queue[ActiveRequest]
    _detokenize_backlog: queue.Queue[tp.Any]
    _decode_slots: queue.Queue[int]
    _active_requests: dict[int, ActiveRequest | None]

    def __init__(
        self,
        engine: vEngine,
        interleaved_mode: bool = False,
        detokenizing_blocks: int = 8,
        slot_clear_steps: int = 512,
        verbose: bool = True,
        prefill_backlog_maxsize: int = 0,
        transfer_backlog_maxsize: int | None = None,
        decode_backlog_maxsize: int | None = None,
        use_operation_lock: bool = True,
        metrics_log_interval_sec: float = 10.0,
    ):
        """
        Initializes the vDriver.

        Args:
            engine: The `vEngine` instance to drive.
            interleaved_mode: If True, operates in a mode that may prioritize
                new requests for latency. Defaults to False.
            detokenizing_blocks: The capacity of the detokenization queue.
                Defaults to 8.
            slot_clear_steps: Number of decode steps after which unused slot
                resources are freed. Defaults to 512.
            verbose: If True, enables informational logging. Defaults to True.
            prefill_backlog_maxsize: Maximum size of the prefill request queue.
                0 means unbounded. Defaults to 0.
            transfer_backlog_maxsize: Maximum size of the KV cache transfer queue.
                If None, uses a default based on `interleaved_mode`.
            decode_backlog_maxsize: Maximum size of the decode request queue.
                If None, uses a default based on `interleaved_mode`.
            use_operation_lock: If True, a lock serializes device-heavy operations
                (prefill, insert, decode) to potentially prevent OOM errors at the
                cost of some parallelism. Defaults to True.
            metrics_log_interval_sec: Interval in seconds for the metrics
                monitor to log aggregated metrics. If 0 or less, periodic
                logging by the monitor is disabled. Defaults to 10.0.
        """
        self._pause = False
        self._engine = engine
        self._interleaved_mode = interleaved_mode
        self._detokenizing_blocks = detokenizing_blocks
        self._slot_clear_steps = slot_clear_steps
        self._use_operation_lock = use_operation_lock

        self.prefill_backlog_maxsize = prefill_backlog_maxsize
        self.transfer_backlog_maxsize = transfer_backlog_maxsize
        self.decode_backlog_maxsize = decode_backlog_maxsize

        self.metrics_recorder = MetricsRecorder(metrics_log_interval_sec)

        if self._use_operation_lock:
            self._operation_lock = threading.Lock()
        else:
            self._operation_lock = None

        self._setup_scheduler()
        self.log = logger.info if verbose else logger.debug
        self.live = False
        self._metrics_thread = None

    def _acquire_operation_lock_if_needed(self):
        """Acquires the operation lock if it's enabled and records wait time."""
        if self._operation_lock:
            start_wait = time.perf_counter()
            self._operation_lock.acquire()
            wait_time_ms = (time.perf_counter() - start_wait) * 1000
            if wait_time_ms > 0.1:
                self.metrics_recorder.record_operation_lock_wait_time(wait_time_ms)

    def _release_operation_lock_if_needed(self):
        """Releases the operation lock if it's enabled."""
        if self._operation_lock:
            self._operation_lock.release()

    @property
    def driver_name(self) -> str:
        """
        Returns a standardized name for the driver and its model.

        The name typically includes the model type and size.

        Returns:
            str: The driver name.
        """
        return self._get_model_name(self._engine.model)

    def place_request_on_prefill_queue(self, request: ActiveRequest):
        """
        Places a new active request onto the prefill backlog queue.

        If the queue is full (and has a max size), this method will block
        until space is available.

        Args:
            request (ActiveRequest): The request to be prefilled.
        """
        try:
            self._prefill_backlog.put(request, block=False)
        except queue.Full:
            self.log(f"Prefill backlog is full (max_size={self.prefill_backlog_maxsize}). Blocking to place request.")
            self._prefill_backlog.put(request, block=True)

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
                    temperature=jnp.array([1], "f4"),
                    top_p=jnp.array([1], "f4"),
                    rngs=engine.prng_key,
                    slot=0,
                )
                decode_state = engine.insert(state_new, decode_state, 0)

            self.log("Compiling decode")
            decode_state = engine.free_state_resources([0], decode_state)
            decode_state = engine.decode(
                graphstate=engine.graphstate,
                graphothers=engine.graphothers,
                state=decode_state,
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
        """
        Returns the total number of concurrent decode requests the engine can handle.

        Returns:
            int: Maximum concurrent decodes supported by the engine.
        """
        return self._engine.total_max_concurrent_decodes

    def _jax_transfer_prefill_result(self, new_request: ActiveRequest):
        """
        Transfers prefill results (KV cache) to the engine's device.

        Uses `jax.device_put` and blocks until the transfer is complete.
        Records the duration of this transfer operation.

        Args:
            new_request (ActiveRequest): The request whose `prefill_result`
                needs to be transferred.
        """
        start_time = time.perf_counter()
        dst_sharding = self._engine.get_prefix_destination_sharding()
        new_request.prefill_result = jax.device_put(new_request.prefill_result, dst_sharding)
        jax.block_until_ready(new_request.prefill_result)
        duration_ms = (time.perf_counter() - start_time) * 1000
        self.metrics_recorder.record_transfer_op_time(duration_ms)

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
                    slot_max_length=request.max_tokens,
                    result_tokens=request_first_token,
                    eos_token_id=engine.eos_token_ids,
                    is_client_side_tokenization=request.is_client_side_tokenization,
                    complete=request.complete,
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
                self.log(f"TTFT: {first_token_return_time:.2f}ms for a request.")

            elif len(data) == 2 and isinstance(data[1], ResultTokens):
                _, result_tokens = data
                result_tokens = result_tokens.convert_to_numpy()

                for slot, request_obj in list(self._live_requests.items()):
                    if request_obj is not None:
                        request: ActiveRequest = request_obj
                        if request.decode_start_time is None:
                            request.decode_start_time = time.perf_counter()

                        (
                            results_base,
                            complete,
                            num_valid_tokens_list,
                        ) = process_result_tokens(
                            processor=processor,
                            slot=slot,
                            slot_max_length=request.max_tokens,
                            result_tokens=result_tokens,
                            eos_token_id=engine.eos_token_ids,
                            is_client_side_tokenization=request.is_client_side_tokenization,
                            complete=request.complete,
                        )
                        request.complete = complete
                        elapsed_time = time.perf_counter() - request.decode_start_time
                        final_step_results = []
                        for res_base, num_valid in zip(results_base, num_valid_tokens_list, strict=False):
                            if len(res_base.text) > 0:
                                for idx, (accum, res) in enumerate(
                                    zip(
                                        request.accumulated_text,
                                        res_base.text,
                                        strict=False,
                                    )
                                ):
                                    request.accumulated_text[idx] = accum + res
                            if request.stop is not None:
                                for stop_sign in request.stop:
                                    for idx, accum in enumerate(request.accumulated_text):
                                        if stop_sign in accum:
                                            request.complete[idx] = True
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
                            self.log(f"Request in slot {slot} completed.")
                            self.metrics_recorder.increment_completed_requests()
                            self._live_requests[slot] = None
                            try:
                                self._decode_slots.put(slot, block=False)
                            except queue.Full:
                                self.log(
                                    f"Decode slots queue full when trying to return slot {slot}. This should not happen."
                                )
                            engine.free_resource(slot)
            elif len(data) == 2 and isinstance(data[1], ActiveRequest):
                slot, active_request = data
                self.log(f"Tracking new active request in slot {slot}.")
                self._live_requests[slot] = active_request
            else:
                self.log(
                    f"Warning: Unknown data type received in detokenize backlog: {type(data[0]) if data else 'None'}"
                )

    def _decode_action_thread(self):
        """
        Background thread action for inserting prefills and performing decode steps.

        This thread consumes requests from `_decode_backlog`, inserts their
        prefill results (KV caches) into the engine's decode state using
        available slots. It then calls the engine's decode method to generate
        the next set of tokens for all active requests. Results are passed to
        the `_detokenize_backlog`.
        Device-heavy operations (`engine.insert`, `engine.decode`) are
        optionally serialized by `_operation_lock`.
        """
        engine = self._engine
        generate_timestep = 0
        decode_state = engine.init_decode_state()
        last_inserted_slot_in_cycle = -1

        while self.live:
            inserted_new_request_this_cycle = False
            last_inserted_slot_in_cycle = -1

            try:
                self._acquire_operation_lock_if_needed()
                while not self._decode_slots.empty() and not self._decode_backlog.empty():
                    try:
                        slot_for_insert = self._decode_slots.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        new_request = self._decode_backlog.get_nowait()
                        if new_request is None:
                            self._decode_slots.put(slot_for_insert, block=False)
                            if not self.live:
                                break
                            continue
                    except queue.Empty:
                        self._decode_slots.put(slot_for_insert, block=False)
                        break

                    self.log(f"Decode filling slot {slot_for_insert} with a new request at step {generate_timestep}.")
                    insert_start_time = time.perf_counter()
                    decode_state = engine.insert(
                        prefix=new_request.prefill_result,
                        decode_state=decode_state,
                        slot=slot_for_insert,
                    )
                    insert_duration_ms = (time.perf_counter() - insert_start_time) * 1000
                    self.metrics_recorder.record_insert_op_time(insert_duration_ms)

                    del new_request.prefill_result
                    new_request.generate_timestep_added = generate_timestep
                    new_request.complete = np.zeros((engine.samples_per_slot,), "b1")
                    self._detokenize_backlog.put((slot_for_insert, new_request), block=True)
                    inserted_new_request_this_cycle = True
                    last_inserted_slot_in_cycle = slot_for_insert
            finally:
                self._release_operation_lock_if_needed()

            if not self.live and (
                self._decode_backlog.empty() or not any(v is not None for v in self._live_requests.values())
            ):
                break

            decode_performed_this_iteration = False
            if any(v is not None for v in self._live_requests.values()):
                time_before_decode_call = time.perf_counter()
                try:
                    self._acquire_operation_lock_if_needed()
                    slot_for_decode_op = last_inserted_slot_in_cycle if last_inserted_slot_in_cycle != -1 else 0

                    temp_decode_state, sampled_tokens = engine.decode(
                        graphstate=engine.graphstate,
                        graphothers=engine.graphothers,
                        state=decode_state,
                        rngs=engine.prng_key,
                        slot=slot_for_decode_op,
                    )
                    decode_op_duration_ms = (time.perf_counter() - time_before_decode_call) * 1000
                    self.metrics_recorder.record_decode_op_time(decode_op_duration_ms)

                    sampled_tokens.copy_to_host_async()
                    decode_state = temp_decode_state
                    self._detokenize_backlog.put((generate_timestep, sampled_tokens), block=True)
                    decode_performed_this_iteration = True
                finally:
                    self._release_operation_lock_if_needed()

                total_decode_cycle_ms = (time.perf_counter() - time_before_decode_call) * 1000
                self.log(
                    f"Decode engine step {generate_timestep} - UsedSlotForDecode: {slot_for_decode_op}, "
                    f"DecodeOpTook: {decode_op_duration_ms:.2f}ms, TotalCycleTook: {total_decode_cycle_ms:.2f}ms"
                )

            if decode_performed_this_iteration:
                generate_timestep += 1
                if (generate_timestep % self._slot_clear_steps) == 0:
                    self.log(f"Decode step {generate_timestep}: Clearing unused slot resources.")
                    try:
                        self._acquire_operation_lock_if_needed()
                        decode_state = engine.free_state_resources(
                            [i for i, v in self._live_requests.items() if v is None],
                            decode_state,
                        )
                    finally:
                        self._release_operation_lock_if_needed()
            elif not inserted_new_request_this_cycle:
                time.sleep(0.005)

    def _prefill_action_thread(self):
        """
        Background thread action for performing prefill operations.

        This thread consumes requests from the `_prefill_backlog`, processes
        their content (tokenization, padding), and calls the engine's prefill
        method. The prefill result (KV cache and first token) is then passed
        to the `_detokenize_backlog` (for the first token) and the main request
        object is put onto the `_transfer_backlog`.
        The device-heavy `engine.prefill` operation is optionally serialized
        by `_operation_lock`.
        """
        engine = self._engine
        processor = engine.processor
        while self.live:
            try:
                request = self._prefill_backlog.get(block=True, timeout=0.1)
            except queue.Empty:
                if not self.live:
                    break
                continue
            if request is None:
                break

            request.metadata.prefill_dequeue_time = time.perf_counter()

            (
                (
                    padded_tokens,
                    padded_valids,
                    true_length,
                ),
                sampling_params,
            ) = self._process_prefill_content(
                request,
                processor,
                engine.max_prefill_length,
                engine.prefill_lengths,
                engine.pad_token_id,
            )
            self.log(f"Prefill for a request: BLSize={self._prefill_backlog.qsize()}, Tokens={padded_valids.shape[-1]}")

            prefill_start_time = time.perf_counter()
            try:
                self._acquire_operation_lock_if_needed()
                prefill_result, first_token = engine.prefill(
                    graphstate=engine.graphstate,
                    graphothers=engine.graphothers,
                    tokens=padded_tokens,
                    valids=padded_valids,
                    true_length=true_length,
                    temperature=jnp.array([sampling_params.temperature], "f4"),
                    top_p=jnp.array([sampling_params.top_p], "f4"),
                    rngs=engine.prng_key,
                    slot=0,
                )
            finally:
                self._release_operation_lock_if_needed()
            prefill_duration_ms = (time.perf_counter() - prefill_start_time) * 1000
            self.metrics_recorder.record_prefill_op_time(prefill_duration_ms)

            request.prefill_result = prefill_result
            request.complete = np.zeros((engine.samples_per_slot,), "b1")
            self._detokenize_backlog.put(
                (first_token, request, request.metadata.prefill_dequeue_time),
                block=True,
            )
            try:
                self._transfer_backlog.put(request, block=True)
                self.log(f"Placed a request on transfer queue ({self._transfer_backlog.qsize()} items).")
            except queue.Full:
                self.log("Transfer backlog full. Blocking to place a request.")
                self._transfer_backlog.put(request, block=True)
            del prefill_result, request

    def _process_prefill_content(
        self,
        request: ActiveRequest,
        processor: ProcessingClassType,
        max_prefill_length: int,
        prefill_lengths: list[int],
        pad_token_id: int,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, int], SamplingParams]:
        """
        Processes the content of a prefill request for the engine.

        This involves tokenizing the input (if it's a string), padding it to
        an appropriate length based on engine capabilities and bucketing,
        and preparing sampling parameters.

        Args:
            request: The `ActiveRequest` containing the prompt and settings.
            processor: The tokenizer/processor instance.
            max_prefill_length: Maximum allowed length for prefill.
            prefill_lengths: List of supported prefill bucket lengths.
            pad_token_id: The ID of the padding token.

        Returns:
            A tuple containing:
                - A nested tuple: (padded_tokens, padded_valids, true_length)
                - The constructed `SamplingParams` object.
        """
        content = request.prefill_content
        if isinstance(content, str):
            content = processor(text=content, return_tensors="np", return_attention_mask=True)
            tokens = jnp.array(content["input_ids"])
            valids = jnp.array(content["attention_mask"])
        else:
            tokens, valids = content

        return (
            pad_tokens(
                tokens=tokens,
                valids=valids,
                pad_token_id=pad_token_id,
                max_prefill_length=max_prefill_length,
                prefill_lengths=prefill_lengths,
                right_padding=False,
            ),
            SamplingParams(
                max_tokens=0,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
                top_p=request.top_p,
                temperature=request.temperature,
            ),
        )

    def _setup_scheduler(self):
        """
        Sets up the internal queues and request tracking structures.

        Queue sizes are determined based on configuration parameters and
        `interleaved_mode`.
        """
        engine = self._engine
        _transfer_q_size = self.transfer_backlog_maxsize
        if _transfer_q_size is None:
            _transfer_q_size = 1 if self._interleaved_mode else 4
        if _transfer_q_size <= 0 and self.transfer_backlog_maxsize is not None:
            _transfer_q_size = 1

        _decode_q_size = self.decode_backlog_maxsize
        if _decode_q_size is None:
            _decode_q_size = 1 if self._interleaved_mode else max(1, engine.max_concurrent_decodes // 3)
        if _decode_q_size <= 0 and self.decode_backlog_maxsize is not None:
            _decode_q_size = 1

        self._prefill_backlog = queue.Queue(
            maxsize=self.prefill_backlog_maxsize if self.prefill_backlog_maxsize > 0 else 0
        )
        self._transfer_backlog = queue.Queue(maxsize=_transfer_q_size)
        self._decode_backlog = queue.Queue(maxsize=_decode_q_size)
        self._detokenize_backlog = queue.Queue(maxsize=self._detokenizing_blocks if self._detokenizing_blocks > 0 else 0)
        self._decode_slots = queue.Queue(maxsize=engine.max_concurrent_decodes)
        self._live_requests: dict[int, ActiveRequest | None] = {i: None for i in range(engine.max_concurrent_decodes)}
        for i in range(engine.max_concurrent_decodes):
            self._decode_slots.put(i)

        self._prefill_thread = SafeThread(target=self._prefill_action_thread, name="prefill-thread", daemon=True)
        self._transfer_thread = SafeThread(target=self._transfer_action_thread, name="transfer-thread", daemon=True)
        self._decode_thread = SafeThread(target=self._decode_action_thread, name="decode-thread", daemon=True)
        self._detokenize_thread = SafeThread(
            target=self._detokenize_action_thread, name="detokenize-thread", daemon=True
        )

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
                self.metrics_recorder.update_queue_size("prefill_backlog", self._prefill_backlog.qsize())
                self.metrics_recorder.update_queue_size("transfer_backlog", self._transfer_backlog.qsize())
                self.metrics_recorder.update_queue_size("decode_backlog", self._decode_backlog.qsize())
                self.metrics_recorder.update_queue_size("detokenize_backlog", self._detokenize_backlog.qsize())
                self.metrics_recorder.update_queue_size("free_decode_slots", self._decode_slots.qsize())

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
                        "lock_wait_ms_avg": aggregated_metrics.get("operation_lock_wait_ms_avg"),
                        "prefill_ops_total": aggregated_metrics.get("prefill_ops_count"),
                        "decode_ops_total": aggregated_metrics.get("decode_ops_count"),
                    }
                    log_summary_filtered = {k: v for k, v in log_summary.items() if v is not None}

                    device_stats = self.get_device_memory_stats()
                    if device_stats:
                        log_summary_filtered["device_memory"] = device_stats

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

    def start(self):
        """
        Starts the vDriver and all its background worker threads.

        This includes threads for prefill, transfer, decode, detokenization,
        and metrics monitoring. Sets the `live` flag to True.
        """
        if not self.live:
            self.log("Starting vDriver...")
            self._all_threads = [
                self._prefill_thread,
                self._transfer_thread,
                self._decode_thread,
                self._detokenize_thread,
            ]
            self.live = True
            for t in self._all_threads:
                if not t.is_alive():
                    self.log(f"Starting thread: {t.name}")
                    t.start()

            if self._metrics_thread is None or not self._metrics_thread.is_alive():
                self._metrics_thread = SafeThread(
                    target=self._metrics_monitor_thread_action,
                    name="metrics-monitor-thread",
                    daemon=True,
                )
                self.log("Starting metrics monitor thread.")
                self._metrics_thread.start()
                self._all_threads.append(self._metrics_thread)

            self.log("vDriver started.")

    def stop(self):
        """
        Stops the vDriver and all its background worker threads gracefully.

        Sets the `live` flag to False, signals threads to terminate by putting
        `None` sentinels in queues, and attempts to join all threads.
        Orphaned requests in queues have their return channels closed.
        """
        if self.live:
            self.log("Stopping vDriver...")
            self.live = False

            queues_to_signal = [
                self._prefill_backlog,
                self._transfer_backlog,
                self._decode_backlog,
                self._detokenize_backlog,
            ]
            for q_safe in queues_to_signal:
                try:
                    q_safe.put_nowait(None)
                except queue.Full:
                    self.log(f"Queue {q_safe} full while trying to send sentinel. May delay shutdown.")

            current_threads_to_join = list(self._all_threads)
            for t in current_threads_to_join:
                if t.is_alive():
                    self.log(f"Joining thread: {t.name}")
                    t.join(timeout=1.0)

            self.log("Draining queues and closing request channels...")
            for q_final_drain in queues_to_signal:
                while True:
                    try:
                        r = q_final_drain.get_nowait()
                        if r is None:
                            continue

                        request_to_close = None
                        if isinstance(r, ActiveRequest):
                            request_to_close = r
                        elif isinstance(r, tuple) and r:
                            if isinstance(r[0], ActiveRequest):
                                request_to_close = r[0]
                            elif len(r) > 1 and isinstance(r[1], ActiveRequest):
                                request_to_close = r[1]

                        if request_to_close and request_to_close.return_channel:
                            self.log("Closing return channel for an orphaned request.")
                            request_to_close.return_channel.close()
                            request_to_close.return_channel = None
                    except queue.Empty:
                        break

            for t_final in current_threads_to_join:
                if t_final.is_alive():
                    self.log(f"Thread {t_final.name} still alive after initial join. Attempting longer join...")
                    t_final.join(timeout=3.0)
                    if t_final.is_alive():
                        self.log(f"ERROR: Thread {t_final.name} FAILED to terminate.")
            self.log("vDriver stopped.")
            if self._metrics_thread and self._metrics_thread.is_alive():
                self.log("Metrics thread still alive after main stop, forcing join.")
                self._metrics_thread.join(timeout=1.0)

    def resume(self):
        """
        Resumes the vDriver if it was previously paused.

        Re-initializes the scheduler queues and starts all worker threads.
        The metrics recorder continues accumulating from its previous state.
        """
        if self._pause:
            self.log("Resuming vDriver...")
            self._setup_scheduler()
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

    def submit_request(self, request: tp.Any):
        """
        Submits a new request for processing by the vDriver.

        The request is placed on the prefill queue.

        Args:
            request (tp.Any): The request to submit. Must be an instance
                of `ActiveRequest`.

        Raises:
            TypeError: If the submitted request is not an `ActiveRequest`.
        """
        if not isinstance(request, ActiveRequest):
            raise TypeError("Request must be of type ActiveRequest")
        if not self.live and not self._pause:
            self.log("Warning: Driver is not live and not paused. Request may not be processed.")
        self.log("Submitting a new request.")
        self.metrics_recorder.increment_submitted_requests()
        self.place_request_on_prefill_queue(request)

    def _transfer_prefill_result(self, new_request: ActiveRequest):
        """
        Helper method to invoke the JAX-specific KV cache transfer.

        Args:
            new_request (ActiveRequest): The request whose prefill result
                (KV cache) needs to be transferred.
        """
        self._jax_transfer_prefill_result(new_request)

    def _transfer_action_thread(self):
        """
        Background thread action for transferring prefill results (KV caches).

        This thread consumes requests from the `_transfer_backlog`. If not in
        `interleaved_mode`, it explicitly transfers the KV cache to the
        engine's device (optionally guarded by `_operation_lock`). The request
        is then placed on the `_decode_backlog`.
        """
        while self.live:
            try:
                new_request = self._transfer_backlog.get(block=True, timeout=0.1)
            except queue.Empty:
                if not self.live:
                    break
                continue
            if new_request is None:
                break

            self.log("Transferring a request.")
            if not self._interleaved_mode:
                self.log("Transferring prefill for a request to Decode engine (interleaved_mode=False).")
                try:
                    self._acquire_operation_lock_if_needed()
                    self._transfer_prefill_result(new_request)
                finally:
                    self._release_operation_lock_if_needed()

            current_decode_backlog_size = self._decode_backlog.qsize()
            try:
                self._decode_backlog.put(new_request, block=True)
                self.log(
                    f"Successfully transferred a request to Decode engine ({current_decode_backlog_size + 1} "
                    "requests now in decode backlog)."
                )
            except queue.Full:
                self.log("Decode backlog full during transfer for a request. Blocking.")
                self._decode_backlog.put(new_request, block=True)

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
            return self.metrics_recorder.get_aggregated_metrics_snapshot(window_size)
        else:
            return self.metrics_recorder.get_all_metrics()

    def get_device_memory_stats(self) -> dict | None:
        """
        Attempts to retrieve memory statistics for the primary local JAX device.

        This relies on the `memory_stats()` method of the JAX device object,
        which may not be available on all platforms or JAX versions.

        Returns:
            tp.Optional[dict]: A dictionary with memory statistics (e.g.,
                'bytes_used', 'bytes_limit') if available, otherwise None.
        """
        try:
            local_devs = jax.local_devices()
            if not local_devs:
                self.log("No local JAX devices found.")
                return None

            device = local_devs[0]
            if hasattr(device, "memory_stats"):
                stats = device.memory_stats()
                return {
                    "device_id": device.id,
                    "platform": device.platform,
                    "device_kind": device.device_kind,
                    "bytes_limit": stats.get("bytes_limit"),
                    "bytes_used": stats.get("bytes_used"),
                    "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
                    "bytes_available": stats.get(
                        "bytes_available",
                        stats.get("bytes_limit", 0) - stats.get("bytes_used", 0),
                    ),
                }
            else:
                self.log(f"Device {device.id} ({device.device_kind}) object does not have memory_stats method.")
                return None
        except Exception as e:
            self.log(f"Could not retrieve JAX device memory stats: {e}")
            return None
