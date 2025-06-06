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


class vDriver:
    """Drives the engines.

    The `vDriver` class manages the prefill and decode engines (`vEngine`),
    orchestrates the data transfer between them, and handles tokenization and
    detokenization. It uses background threads to perform these tasks concurrently,
    allowing for high-throughput inference.

    Attributes:
        _engine (vEngine): vEngine used for the prefill/decode stage.
        _prefill_backlogs (queue.Queue[ActiveRequest | None]): Queue for incoming prefill requests.
        _transfer_backlogs (list[queue.Queue[ActiveRequest]]): Queues for transferring requests from
            prefill to decode engines.
        _decode_backlogs (dict[int, queue.Queue[ActiveRequest]]): Queues for incoming decode requests,
            organized by engine index.
        _detokenize_backlogs (list[queue.Queue[ResultTokens]]): Queues for detokenizing results, organized
            by engine index.
        _decode_slots (list[queue.Queue[int]]): Queues for managing available decode slots in each engine.
        _active_requests (list[queue.Queue[tuple[int, ActiveRequest]]]): Queues for tracking active requests,
            organized by engine index.
        _interleaved_mode (bool): Flag indicating whether interleaved mode is enabled.
        _detokenizing_blocks (int): Number of detokenizing blocks.

    """

    _engine: vEngine
    _prefill_backlog: queue.Queue[ActiveRequest | None]
    _transfer_backlog: queue.Queue[ActiveRequest]
    _decode_backlog: queue.Queue[ActiveRequest]
    _detokenize_backlog: queue.Queue[ResultTokens]
    _decode_slots: queue.Queue[int]
    _active_requests: list[queue.Queue[tuple[int, ActiveRequest]]]
    _interleaved_mode: bool = False
    _slot_clear_steps: int = 512
    _detokenizing_blocks: int = 8

    def __init__(
        self,
        engine: vEngine,
        interleaved_mode: bool = False,
        detokenizing_blocks: int = 8,
        slot_clear_steps: int = 512,
        verbose: bool = True,
    ):
        """Initializes the `vDriver`.

        Sets up the prefill and decode engines, backlogs (queues) for managing
        requests between stages, available slots for concurrent decoding, and
        starts the background threads for each stage (prefill, transfer, decode,
        detokenize).

        Args:
            engine: `vEngine`s to be used for the prefill/decode stage.
            interleaved_mode: A boolean flag indicating whether the driver should
                operate in interleaved mode (potentially optimizing for latency
                by prioritizing new requests). Defaults to False.
            slot_clear_steps: num steps to clear unused slots to free comp-chunk.
            detokenizing_blocks: The number of detokenizing blocks. Defaults to 8.
            verbose: Whether to log information. Defaults to True.
        """

        self._pause = False
        self._engine = engine
        self._interleaved_mode = interleaved_mode
        self._detokenizing_blocks = detokenizing_blocks
        self._slot_clear_steps = slot_clear_steps
        self._setup_scheduler()

        self.log = logger.info if verbose else logger.debug
        self.live = False

    @property
    def driver_name(self):
        """Returns the driver name."""
        return self._get_model_name(self._engine.model)

    def place_request_on_prefill_queue(self, request: ActiveRequest):
        """Used to place new requests for prefilling and generation."""
        self._prefill_backlog.put(request, block=False)

    @property
    def processor(self) -> ProcessingClassType:  # type:ignore
        """Returns the processor/tokenizer associated with the engines.

        Assumes all engines (prefill and decode) use the same processor.
        Raises an error if no engines are configured.
        """
        return self._engine.processor

    def _calculate_model_size(self, graphstate) -> str:
        """
        Calculate model size in billions of parameters.
        Returns formatted string with 2 decimal places.
        """
        try:
            num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
            size_in_billions = num_params / 1e9
            return f"{size_in_billions:.2f}"
        except Exception:
            return "unknown"

    def _get_model_name(self, model) -> str:
        """
        Generate a standardized vsurge name combining model type, size, and timestamp.

        Format: {model_type}-{size_in_B}b
        Example: llama-7.00b
        """
        model_type = self._get_model_type(model)
        model_size = self._calculate_model_size(model.graphstate)

        return f"{model_type}-{model_size}b"

    def _get_model_type(self, model) -> str:
        """Get the model type, with fallback to 'unknown' if not found."""
        return getattr(model.config, "model_type", "unknown").lower()

    def compile(self):
        """Compiles engines."""
        engine = self._engine
        try:
            decode_state = engine.init_decode_state()
            for length in engine.prefill_lengths:
                padded_tokens = padded_valids = jnp.ones((1, length), "i4")
                logger.info(f"Compiling prefill/insert length={length}")
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

            logger.info("Compiling decode")
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
        """Gets the total number of concurrent requests the driver can handle."""
        return self._engine.total_max_concurrent_decodes

    def _jax_transfer_prefill_result(self, new_request: ActiveRequest):
        """Transfers prefill result (KV cache) using JAX device placement.

        This method uses JAX's `jax.device_put` to transfer the prefill result
        (which typically contains the KV cache state after the prefill step)
        to the specified target decode engine's device, respecting its sharding
        configuration. It blocks until the transfer is complete.

        Args:
            new_request: The ActiveRequest containing the prefill_result.
        """
        dst_sharding = self._engine.get_prefix_destination_sharding()
        new_request.prefill_result = jax.device_put(new_request.prefill_result, dst_sharding)
        jax.block_until_ready(new_request.prefill_result)

    def _detokenize_action_thread(self):
        """Detokenize sampled tokens and returns them to the user."""

        engine = self._engine
        processor = engine.processor
        while self.live:
            data = self._detokenize_backlog.get(block=True)
            if data is None:
                break
            if isinstance(data[0], ResultTokens):
                request_first_token, request, _ = data
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

                request.enqueue_samples(final_results)

                first_token_return_time = (time.perf_counter() - request.metadata.prefill_dequeue_time) * 1000
                self.log(f"TTFT duration: {first_token_return_time}ms")

            elif isinstance(data[1], ResultTokens):
                generate_timestep_added, result_tokens = data
                result_tokens = result_tokens.convert_to_numpy()

                for slot, request in self._live_requests.items():
                    if request is not None:
                        request: ActiveRequest = request
                        if request.decode_start_time is None:
                            request.decode_start_time = time.perf_counter()

                        results_base, complete, num_valid_tokens_list = process_result_tokens(
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
                                    zip(request.accumulated_text, res_base.text, strict=False)
                                ):
                                    request.accumulated_text[idx] = accum + res
                            if request.stop is not None:
                                for stop_sign in request.stop:
                                    for idx, accum in enumerate(request.accumulated_text):
                                        if stop_sign in accum:
                                            request.complete[idx] = True
                            request.total_generated_tokens += num_valid
                            tps = (
                                request.total_generated_tokens / elapsed_time
                                if elapsed_time > 1e-6  # Avoid division by zero
                                else 0.0
                            )
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

                        request.enqueue_samples(final_step_results)

                        if request.complete.all():
                            request.metadata.complete_time = time.perf_counter()
                            request.return_channel.close()
                            self._live_requests[slot] = None
                            self._decode_slots.put(slot, block=False)
                            engine.free_resource(slot)
            else:
                slot, active_request = data
                self._live_requests[slot] = active_request

    def _decode_action_thread(self):
        """Step token generation and insert prefills from backlog."""

        engine = self._engine
        generate_timestep = 0
        decode_state = engine.init_decode_state()
        time_of_last_print = time.time()

        while self.live:
            if (time.time() - time_of_last_print) > 5:
                self.log(
                    "Decode thread making a decision with:"
                    f" prefill_backlog={self._prefill_backlog.qsize()}"
                    f" generate_free_slots={self._decode_slots.qsize()}",
                )
                time_of_last_print = time.time()

            max_concurrent_decodes = engine.max_concurrent_decodes
            while True:
                my_slots_size = self._decode_slots.qsize()

                try:
                    slot = self._decode_slots.get(block=False)
                except queue.Empty:
                    break

                block = my_slots_size == max_concurrent_decodes
                if self._interleaved_mode:
                    block |= not self._prefill_backlog.empty()
                    block |= not self._transfer_backlog.empty()
                try:
                    new_request = self._decode_backlog.get(block=block, timeout=1.0)
                    if new_request is None:
                        break
                    new_request.metadata.generate_dequeue_time = time.perf_counter()
                except queue.Empty:
                    self._decode_slots.put(slot, block=False)
                    if block:
                        continue
                    else:
                        break

                if new_request is None:
                    return

                self.log(f"Decode filling slot {slot} at step {generate_timestep}.")

                decode_state = engine.insert(
                    prefix=new_request.prefill_result,
                    decode_state=decode_state,
                    slot=slot,
                )
                del new_request.prefill_result
                new_request.generate_timestep_added = generate_timestep
                new_request.complete = np.zeros((engine.samples_per_slot,), "b1")
                self._detokenize_backlog.put((slot, new_request), block=True)

            assert self._decode_slots.qsize() < max_concurrent_decodes, (
                "At this point we must have some requests inserted into the slots."
            )
            time_of_last_decode = time.time()
            decode_state, sampled_tokens = engine.decode(
                graphstate=engine.graphstate,
                graphothers=engine.graphothers,
                state=decode_state,
                rngs=engine.prng_key,
                slot=slot,
            )
            fn_call = time.time()
            sampled_tokens.copy_to_host_async()
            if ((generate_timestep + 1) % self._slot_clear_steps) == 0:
                decode_state = engine.free_state_resources(
                    [i for i, v in self._live_requests.items() if v is None],
                    decode_state,
                )
            self._detokenize_backlog.put((generate_timestep, sampled_tokens), block=True)
            generate_timestep += 1

            _took = (time.time() - time_of_last_decode) * 10**3
            _exec = (fn_call - time_of_last_decode) * 10**3

            self.log(
                f"Decode engine step {generate_timestep} - slots free : {my_slots_size} / {max_concurrent_decodes}, "
                f"took {_took:.2f}ms  | execution took {_exec:.2f}ms "
            )

    def _prefill_action_thread(self):
        """Thread which runs in the background performing prefills."""
        engine = self._engine
        processor = engine.processor
        while self.live:
            request = self._prefill_backlog.get(block=True)
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
            self.log(f"prefill queue size : {self._prefill_backlog.qsize()}, Token size {padded_valids.shape[-1]}")
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
            request.prefill_result = prefill_result
            request.complete = np.zeros((engine.samples_per_slot,), "b1")

            request.metadata.transfer_enqueue_time = time.perf_counter()
            self._detokenize_backlog.put((first_token, request, request.metadata.prefill_dequeue_time), block=True)
            self._transfer_backlog.put(request, block=True)
            self.log(f"Placed request on transfer queue, {self._transfer_backlog.qsize()} queued requests.")

            del prefill_result
            del request

    def _process_prefill_content(
        self,
        request: ActiveRequest,
        processor: ProcessingClassType,  # type:ignore
        max_prefill_length: int,
        prefill_lengths: list[int],
        pad_token_id: int,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, int], SamplingParams]:
        """Tokenizes, pads, and prepares sampling parameters for a prefill request.

        Takes an `ActiveRequest`, extracts its `prefill_content` (which can be
        a string or pre-tokenized IDs), tokenizes it using the provided
        `processor` if necessary, pads the tokens to the appropriate length
        based on `max_prefill_length` and internal buckets, and constructs
        the `SamplingParams` object from the request's parameters.

        Args:
            request: The ActiveRequest containing the prompt and sampling settings.
            processor: The tokenizer/processor instance.
            max_prefill_length: The maximum allowed length for the prefill sequence.

        Returns:
            A tuple containing:
                - A nested tuple: (padded_tokens, padded_valids, padded_length)
                - The constructed SamplingParams object.
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

    def _ray_transfer_prefill_result(self, new_request: ActiveRequest):
        """Transfers prefill result (KV cache) using Ray's transfer mechanism (if applicable).

        This method is a placeholder for potential future integration with Ray
        or other distributed computing frameworks that provide explicit data
        transfer mechanisms between workers or devices. It assumes the target
        decode engine has a `transfer` method.

        Args:
            new_request: The ActiveRequest containing the prefill_result.
        """
        self._engine.transfer(new_request.prefill_result)

    def _setup_scheduler(self):
        """Sets up the scheduler."""
        engine = self._engine
        _ini_size = 1 if self._interleaved_mode else engine.max_concurrent_decodes // 3

        self._prefill_backlog = queue.Queue()
        self._transfer_backlog = queue.Queue(1 if self._interleaved_mode else 4)
        self._decode_backlog = queue.Queue(_ini_size)
        self._detokenize_backlog = queue.Queue(self._detokenizing_blocks)
        self._decode_slots = queue.Queue(engine.max_concurrent_decodes)
        self._decode_lists = list(range(engine.max_concurrent_decodes))
        self._live_requests = {i: None for i in range(engine.max_concurrent_decodes)}
        _ = [self._decode_slots.put(i) for i in self._decode_lists]

        self._prefill_thread = SafeThread(target=self._prefill_action_thread, name="prefill-thread", daemon=True)
        self._transfer_thread = SafeThread(target=self._transfer_action_thread, name="transfer-thread", daemon=True)
        self._decode_thread = SafeThread(target=self._decode_action_thread, name="decode-thread", daemon=True)
        self._detokenize_thread = SafeThread(target=self._detokenize_action_thread, name="detokenize-thread")

    def replace_graphstate(self, state):
        """Replaces the graph state of the driver."""
        self._engine.graphstate = state

    def start(self):
        """Starts the driver and its associated background processes."""
        if not self.live:
            self._all_threads = [
                self._prefill_thread,
                self._transfer_thread,
                self._decode_thread,
                self._detokenize_thread,
            ]
            self.live = True
            for t in self._all_threads:
                t.start()

    def stop(self):
        """Stops the driver and all background threads."""
        if self.live:
            self.live = False

            all_backlogs = [
                self._prefill_backlog,
                self._transfer_backlog,
                self._decode_backlog,
                self._detokenize_backlog,
            ]

            while any(t.is_alive() for t in self._all_threads):
                for q in all_backlogs:
                    while True:
                        try:
                            r = q.get_nowait()
                            if r is None:
                                continue
                            elif isinstance(r, ActiveRequest):
                                r.return_channel = None
                            else:  # detokenize backlog
                                _, r = r
                                if isinstance(r, ActiveRequest):
                                    r.return_channel = None
                        except queue.Empty:
                            break

                for q in all_backlogs:
                    try:
                        q.put_nowait(None)
                    except queue.Full:
                        pass
            for t in self._all_threads:
                t.join()

    def resume(self):
        """Resumes the driver."""
        if self._pause:
            self._setup_scheduler()
            self.start()
            self._pause = False

    def pause(self):
        """Pauses the driver."""
        if not self._pause:
            self.stop()
            self._pause = True

    def submit_request(self, request: tp.Any):
        """Submits a new request to the driver's processing queue."""
        if not isinstance(request, ActiveRequest):
            raise TypeError("Request must be of type ActiveRequest")
        self.place_request_on_prefill_queue(request)

    def _transfer_prefill_result(self, new_request: ActiveRequest):
        """Selects and executes the appropriate KV cache transfer method.

        This method acts as a dispatcher for transferring the prefill result
        (KV cache) from the prefill engine's device to the target decode
        engine's device. It currently defaults to using the JAX-specific
        transfer method but can be extended to support other frameworks
        like Ray by adding conditional logic based on the engine type or
        configuration.

        Args:
            new_request: The ActiveRequest containing the prefill_result.
            target_idx: The index of the target decode engine.
        """
        self._jax_transfer_prefill_result(new_request)

    def _transfer_action_thread(self):
        """Transfers the kv cache on an active request to the least full
        generate backlog."""

        while self.live:
            new_request = self._transfer_backlog.get(block=True)
            if new_request is None:
                break
            new_request.metadata.transfer_dequeue_time = time.perf_counter()
            if not self._interleaved_mode:
                self.log("Transferring prefill to Decode engine.")
                self._transfer_prefill_result(new_request)
            new_request.metadata.generate_enqueue_time = time.perf_counter()
            _requested_slots = self._decode_backlog.qsize()
            self._decode_backlog.put(new_request, block=True)
            self.log(f"Successfully transferred prefill to Decode engine ({_requested_slots} requests now in backlog).")
