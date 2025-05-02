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

import functools
import itertools
import queue
import time
import traceback
import typing as tp

import jax
import numpy as np
from jax import numpy as jnp

from easydel.inference.utilities import SamplingParams
from easydel.inference.vsurge.engines._abstract_driver import (
	AbstractDriver,
	ProcessingClassType,
)
from easydel.utils.helpers import get_logger

from ...utils import (
	ActiveRequest,
	ReturnSample,
	SafeThread,
	pad_tokens,
	process_result_tokens,
)
from .._utils import ResultTokens
from .engine import vEngine

if tp.TYPE_CHECKING:
	from easydel.infra.utils import ProcessingClassType
else:
	ProcessingClassType = tp.Any


logger = get_logger("vSurge-vDriver")


class vDriver(AbstractDriver):
	"""Drives the engines."""

	_prefill_engines: list[vEngine]
	_decode_engines: list[vEngine]
	_prefill_backlog: queue.Queue[ActiveRequest | None]
	_transfer_backlogs: list[queue.Queue[ActiveRequest]] = []
	_decode_backlogs: dict[int, queue.Queue[ActiveRequest]] = {}
	_detokenize_backlogs: list[queue.Queue[ResultTokens]] = []
	_decode_slots: list[queue.Queue[int]] = []
	_active_requests: list[queue.Queue[tuple[int, ActiveRequest]]] = []
	_interleaved_mode: bool = False
	_detokenizing_blocks: int = 8

	def __init__(
		self,
		prefill_engines: tp.Optional[list[vEngine] | vEngine] = None,
		decode_engines: tp.Optional[list[vEngine] | vEngine] = None,
		interleaved_mode: bool = False,
		detokenizing_blocks: int = 8,
	):
		"""Initializes the vDriver.

		Sets up the prefill and decode engines, backlogs (queues) for managing
		requests between stages, available slots for concurrent decoding, and
		starts the background threads for each stage (prefill, transfer, decode,
		detokenize).

		Args:
		    prefill_engines: A single vEngine or a list of vEngines to be used
		        for the prefill stage. Defaults to an empty list.
		    decode_engines: A single vEngine or a list of vEngines to be used
		        for the decode stage. Defaults to an empty list.
		    interleaved_mode: A boolean flag indicating whether the driver should
		        operate in interleaved mode (potentially optimizing for latency
		        by prioritizing new requests). Defaults to False.
		"""
		if prefill_engines is None:
			prefill_engines = []
		if decode_engines is None:
			decode_engines = []

		if not isinstance(prefill_engines, list):
			prefill_engines = [prefill_engines]

		if not isinstance(decode_engines, list):
			decode_engines = [decode_engines]

		self._prefill_engines = prefill_engines
		self._decode_engines = decode_engines
		self._interleaved_mode = interleaved_mode
		self._detokenizing_blocks = detokenizing_blocks
		self._prefill_backlog = queue.Queue()

		self._transfer_backlogs = [
			queue.Queue(1 if self._interleaved_mode else 4)
			for i in range(len(self._prefill_engines))
		]
		self._decode_backlogs = {
			idx: queue.Queue(
				1 if self._interleaved_mode else engine.max_concurrent_decodes // 3
			)
			for idx, engine in enumerate(self._decode_engines)
		}
		self._detokenize_backlogs = [
			queue.Queue(detokenizing_blocks) for _ in self._decode_engines
		]
		self._decode_slots = [
			queue.Queue(engine.max_concurrent_decodes) for engine in self._decode_engines
		]
		_ = [
			[self._decode_slots[idx].put(i) for i in range(engine.max_concurrent_decodes)]
			for idx, engine in enumerate(self._decode_engines)
		]

		self._prefill_threads = [
			SafeThread(
				target=functools.partial(self._prefill_thread, idx),
				name=f"prefill-{idx}",
				daemon=True,
			)
			for idx in range(len(self._prefill_engines))
		]
		self._transfer_threads = [
			SafeThread(
				target=functools.partial(
					self._transfer_thread,
					idx,
				),
				name=f"transfer-{idx}",
				daemon=True,
			)
			for idx in range(len(self._prefill_engines))
		]
		self._decode_threads = [
			SafeThread(
				target=functools.partial(
					self._decode_thread,
					idx,
				),
				name=f"decode-{idx}",
				daemon=True,
			)
			for idx in range(len(self._decode_engines))
		]
		self.detokenize_threads = [
			SafeThread(
				target=functools.partial(
					self._detokenize_thread,
					idx,
				),
				name=f"detokenize-{idx}",
			)
			for idx in range(len(self._decode_engines))
		]

		self.live = False

	def start(self):
		if not self.live:
			self._all_threads = list(
				itertools.chain(
					self._prefill_threads,
					self._transfer_threads,
					self._decode_threads,
					self.detokenize_threads,
				)
			)
			self.live = True
			for t in self._all_threads:
				t.start()

	# Add this method within the vDriver class
	def submit_request(self, request: tp.Any):
		"""Submits a new request to the driver's processing queue."""
		# Assuming ActiveRequest is the expected type internally
		if not isinstance(request, ActiveRequest):
			# Or raise a more specific error
			raise TypeError("Request must be of type ActiveRequest")
		self.place_request_on_prefill_queue(request)

	@property
	def driver_name(self):
		return self._get_model_name(self._decode_engines[-1].model)

	def compile(self):
		"""Compiles engines."""
		try:
			for (
				decode_engine,
				prefill_engine,
			) in zip(
				self._decode_engines,
				self._prefill_engines,
			):
				decode_state = decode_engine.init_decode_state()
				max_prefill_length = prefill_engine.max_prefill_length
				vals = prefill_engine.prefill_lengths[
					: prefill_engine.prefill_lengths.index(max_prefill_length)
				] + [max_prefill_length]
				for length in vals:
					padded_tokens = padded_valids = jnp.ones((1, length), "i4")
					logger.info(f"Compiling prefill-engine seqlen={length}")
					state_new, _ = prefill_engine.prefill(
						graphstate=prefill_engine.graphstate,
						graphothers=prefill_engine.graphothers,
						tokens=padded_tokens,
						valids=padded_valids,
						true_length=0,
						temperature=jnp.array([1], "f4"),
						top_p=jnp.array([1], "f4"),
						rngs=prefill_engine.prng_key,
					)
					logger.info(f"Compiling decode-engine insert seqlen={length}")
					decode_state = decode_engine.insert(state_new, decode_state, 0)

				logger.info("Compiling decode-engine")
				decode_engine.decode(
					graphstate=decode_engine.graphstate,
					graphothers=decode_engine.graphothers,
					state=decode_state,
					rngs=decode_engine.prng_key,
				)
		except Exception:
			traceback.print_exc()
			self.stop()
			exit(1)

	def stop(self):
		"""Stops the driver and all background threads."""
		if self.live:
			self.live = False

			all_backlogs = list(
				itertools.chain(
					[self._prefill_backlog],
					self._transfer_backlogs,
					self._decode_backlogs.values(),
					self._detokenize_backlogs,
				)
			)

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

	def get_total_concurrent_requests(self) -> int:
		"""Gets the total number of concurrent requests the driver can handle."""
		total_max_concurrent_decodes = sum(
			[e.max_concurrent_decodes for e in self._decode_engines]
		)
		return total_max_concurrent_decodes

	def place_request_on_prefill_queue(self, request: ActiveRequest):
		"""Used to place new requests for prefilling and generation."""
		self._prefill_backlog.put(request, block=False)

	@property
	def processor(self) -> ProcessingClassType:  # type:ignore
		"""Returns the processor/tokenizer associated with the engines.

		Assumes all engines (prefill and decode) use the same processor.
		Raises an error if no engines are configured.
		"""
		if self._prefill_engines:
			return self._prefill_engines[0].processor
		elif self._decode_engines:
			return self._decode_engines[0].processor
		else:
			raise ValueError(
				"No engines configured for the vDriver, cannot determine processor."
			)

	def _process_prefill_content(
		self,
		request: ActiveRequest,
		processor: ProcessingClassType,  # type:ignore
		max_prefill_length: int,
		prefill_lengths: list[int],
		pad_token_id: int,
	) -> tp.Tuple[tp.Tuple[jnp.ndarray, jnp.ndarray, int], SamplingParams]:
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

	def _prefill_thread(self, idx: int):
		"""Thread which runs in the background performing prefills."""
		logger.info(f"Spinning up prefill thread {idx}.")
		prefill_engine = self._prefill_engines[idx]
		processor = prefill_engine.processor

		while self.live:
			my_transfer_backlog = self._transfer_backlogs[idx]
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
				prefill_engine.max_prefill_length,
				prefill_engine.prefill_lengths,
				prefill_engine.pad_token_id,
			)
			logger.info(
				f"Prefilling on prefill engine {idx} : "
				f"prefill queue size : {self._prefill_backlog.qsize()}, Token size {padded_valids.shape[-1]}",
			)
			prefill_result, first_token = prefill_engine.prefill(
				graphstate=prefill_engine.graphstate,
				graphothers=prefill_engine.graphothers,
				tokens=padded_tokens,
				valids=padded_valids,
				true_length=true_length,
				temperature=jnp.array([sampling_params.temperature], "f4"),
				top_p=jnp.array([sampling_params.top_p], "f4"),
				rngs=prefill_engine.prng_key,
			)
			request.prefill_result = prefill_result
			request.complete = np.zeros((prefill_engine.samples_per_slot,), "b1")
			my_detokenize_backlog = self._detokenize_backlogs[idx]
			request.metadata.transfer_enqueue_time = time.perf_counter()
			my_detokenize_backlog.put(
				(first_token, request, request.metadata.prefill_dequeue_time),
				block=True,
			)

			my_transfer_backlog.put(request, block=True)
			logger.info(
				f"Placed request on transfer queue {idx}, {my_transfer_backlog.qsize()} queued requests.",
			)

			del prefill_result
			del request

	def _jax_transfer_prefill_result(self, new_request: ActiveRequest, target_idx: int):
		"""Transfers prefill result (KV cache) using JAX device placement.

		This method uses JAX's `jax.device_put` to transfer the prefill result
		(which typically contains the KV cache state after the prefill step)
		to the specified target decode engine's device, respecting its sharding
		configuration. It blocks until the transfer is complete.

		Args:
			   new_request: The ActiveRequest containing the prefill_result.
			   target_idx: The index of the target decode engine.
		"""
		new_request.prefill_result = jax.device_put(
			new_request.prefill_result,
			self._decode_engines[target_idx].get_prefix_destination_sharding(),
		)
		jax.block_until_ready(new_request.prefill_result)

	def _ray_transfer_prefill_result(self, new_request: ActiveRequest, target_idx: int):
		"""Transfers prefill result (KV cache) using Ray's transfer mechanism (if applicable).

		This method is a placeholder for potential future integration with Ray
		or other distributed computing frameworks that provide explicit data
		transfer mechanisms between workers or devices. It assumes the target
		decode engine has a `transfer` method.

		Args:
		    new_request: The ActiveRequest containing the prefill_result.
		    target_idx: The index of the target decode engine.
		"""
		# Assuming self._decode_engines[target_idx] has a 'transfer' method for Ray
		self._decode_engines[target_idx].transfer(new_request.prefill_result)

	def _transfer_prefill_result(self, new_request: ActiveRequest, target_idx: int):
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
		self._jax_transfer_prefill_result(new_request, target_idx)

	def _transfer_thread(self, idx: int):
		"""Transfers the kv cache on an active request to the least full
		generate backlog."""
		transfer_backlog = self._transfer_backlogs[idx]

		while self.live:
			new_request = transfer_backlog.get(block=True)
			if new_request is None:
				break
			new_request.metadata.transfer_dequeue_time = time.perf_counter()
			target_idx = min(self._decode_backlogs.items(), key=lambda q: q[1].qsize())[0]
			if not self._interleaved_mode:
				logger.info(
					f"Transferring prefill from prefill engine {idx} to Decode engine {target_idx}."
				)
				self._transfer_prefill_result(new_request, target_idx)
			new_request.metadata.generate_enqueue_time = time.perf_counter()
			self._decode_backlogs[target_idx].put(new_request, block=True)
			logger.info(
				"Successfully transferred prefill "
				f"from prefill engine {idx} to Decode engine {target_idx} "
				f"({self._decode_backlogs[target_idx].qsize()} requests now in backlog).",
			)

	def _decode_thread(self, idx: int):
		"""Step token generation and insert prefills from backlog."""
		logger.info(f"Spinning up decode thread {idx}.")

		decode_engine = self._decode_engines[idx]
		my_slots = self._decode_slots[idx]
		my_decode_backlog = self._decode_backlogs[idx]
		my_detokenize_backlog = self._detokenize_backlogs[idx]

		generate_timestep = 0

		decode_state = decode_engine.init_decode_state()

		time_of_last_print = time.time()
		while self.live:
			if (time.time() - time_of_last_print) > 5:
				logger.info(
					"Decode thread making a decision with:"
					f" prefill_backlog={self._prefill_backlog.qsize()}"
					f" generate_free_slots={my_slots.qsize()}",
				)
				time_of_last_print = time.time()

			max_concurrent_decodes = decode_engine.max_concurrent_decodes
			while True:
				my_slots_size = my_slots.qsize()

				try:
					slot = my_slots.get(block=False)
				except queue.Empty:
					break

				block = my_slots_size == max_concurrent_decodes
				if self._interleaved_mode:
					block |= not self._prefill_backlog.empty()
					block |= not self._transfer_backlogs[idx].empty()
				try:
					new_request = my_decode_backlog.get(block=block, timeout=1.0)
					if new_request is None:
						break
					new_request.metadata.generate_dequeue_time = time.perf_counter()
				except queue.Empty:
					my_slots.put(slot, block=False)
					if block:
						continue
					else:
						break

				if new_request is None:
					return

				logger.info(
					f"Decode slice {idx} filling slot {slot} at step {generate_timestep}."
				)

				decode_state = decode_engine.insert(
					prefix=new_request.prefill_result,
					decode_state=decode_state,
					slot=slot,
				)
				del new_request.prefill_result
				new_request.generate_timestep_added = generate_timestep
				new_request.complete = np.zeros((decode_engine.samples_per_slot,), "b1")
				my_detokenize_backlog.put((slot, new_request), block=True)

			assert my_slots.qsize() < max_concurrent_decodes, (
				"At this point we must have some requests inserted into the slots."
			)
			time_of_last_decode = time.time()
			decode_state, sampled_tokens = decode_engine.decode(
				graphstate=decode_engine.graphstate,
				graphothers=decode_engine.graphothers,
				state=decode_state,
				rngs=decode_engine.prng_key,
			)
			fn_call = time.time()
			sampled_tokens.copy_to_host_async()
			my_detokenize_backlog.put((generate_timestep, sampled_tokens), block=True)
			generate_timestep += 1
			# TODO:Debug
			_took = (time.time() - time_of_last_decode) * 10**3
			_exec = (fn_call - time_of_last_decode) * 10**3
			logger.info(
				f"Decode engine {idx} step {generate_timestep} - slots free : {my_slots_size} / {max_concurrent_decodes}, "
				f"took {_took:.2f}ms  | execution took {_exec:.2f}ms "
			)

	def _detokenize_thread(self, idx: int):
		"""Detokenize sampled tokens and returns them to the user."""
		my_detokenize_backlog = self._detokenize_backlogs[idx]
		my_decode_engine = self._decode_engines[idx]
		my_slots = self._decode_slots[idx]

		processor = my_decode_engine.processor
		my_live_requests = {i: None for i in range(my_decode_engine.max_concurrent_decodes)}
		while self.live:
			data = my_detokenize_backlog.get(block=True)
			if data is None:
				break
			if isinstance(data[0], ResultTokens):
				# Handling the very first token from prefill
				request_first_token, request, _ = data
				request_first_token = request_first_token.convert_to_numpy()
				# Process the first token, but TPS/count are not meaningful yet
				results_base, complete, num_valid_tokens_list = process_result_tokens(
					processor=processor,
					slot=0,  # Prefill result is always at slot 0 conceptually
					slot_max_length=request.max_tokens,
					result_tokens=request_first_token,
					eos_token_id=my_decode_engine.eos_token_ids,
					is_client_side_tokenization=request.is_client_side_tokenization,
					complete=request.complete,
				)
				request.complete = complete
				# Add placeholder metrics for the first token
				final_results = []
				for res_base, num_valid in zip(results_base, num_valid_tokens_list):
					# Start tracking total generated tokens from the first valid one
					request.total_generated_tokens += num_valid
					final_results.append(
						ReturnSample(
							text=res_base.text,
							token_ids=res_base.token_ids,
							tokens_per_second=0.0,  # Not applicable yet
							num_generated_tokens=request.total_generated_tokens,
						)
					)

				request.enqueue_samples(final_results)

				first_token_return_time = (
					time.perf_counter() - request.metadata.prefill_dequeue_time
				) * 1000
				logger.info(f"TTFT duration: {first_token_return_time}ms")

			elif isinstance(data[1], ResultTokens):
				# Handling subsequent decode steps
				generate_timestep_added, result_tokens = data
				result_tokens = result_tokens.convert_to_numpy()

				for slot, request in my_live_requests.items():
					if request is not None:
						# Start timer on the first actual decode step for this request
						if request.decode_start_time is None:
							request.decode_start_time = time.perf_counter()

						results_base, complete, num_valid_tokens_list = process_result_tokens(
							processor=processor,
							slot=slot,
							slot_max_length=request.max_tokens,
							result_tokens=result_tokens,
							eos_token_id=my_decode_engine.eos_token_ids,
							is_client_side_tokenization=request.is_client_side_tokenization,
							complete=request.complete,
						)

						request.complete = complete
						elapsed_time = time.perf_counter() - request.decode_start_time
						final_step_results = []

						for res_base, num_valid in zip(results_base, num_valid_tokens_list):
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
									tokens_per_second=tps,
									num_generated_tokens=request.total_generated_tokens,
								)
							)

						request.enqueue_samples(final_step_results)

						if request.complete.all():
							request.metadata.complete_time = time.perf_counter()
							request.return_channel.close()
							my_live_requests[slot] = None
							my_slots.put(slot, block=False)
							my_decode_engine.free_resource(slot)
			else:
				slot, active_request = data
				my_live_requests[slot] = active_request
