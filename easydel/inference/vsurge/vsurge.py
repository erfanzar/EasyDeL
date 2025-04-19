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

import asyncio
import dataclasses
import datetime
import functools
import itertools
import os
import queue
import signal
import threading
import time
import traceback
import typing as tp
from concurrent import futures

import jax
import numpy as np
from jax import numpy as jnp

from easydel.inference.utilities import SamplingParams
from easydel.utils.helpers import get_logger

from .engine import ResultTokens, vEngine
from .utils import (
	ReturnSample,
	is_byte_token,
	pad_tokens,
	process_result_tokens,
	text_tokens_to_string,
)

if tp.TYPE_CHECKING:
	from easydel.infra.base_module import EasyDeLBaseModule
	from easydel.infra.utils import ProcessingClassType
else:
	ProcessingClassType = tp.Any
	EasyDeLBaseModule = tp.Any

logger = get_logger("vSurge")

V = tp.TypeVar("V")


class vSurgeMetadata:
	"""Tracks timing information for requests processed by the vsurge.

	Attributes:
	    start_time: The time when the request processing started.
	"""

	def __init__(self):
		"""Initializes the metadata, capturing the current time as the start time."""
		self.start_time = time.time()


@dataclasses.dataclass
class vSurgeRequest:
	"""Represents a request specifically for text completion."""

	prompt: str
	max_tokens: int
	top_p: float = 1.0
	top_k: int = 0
	min_p: float = 0.0
	temperature: float = 0.0
	presence_penalty: float = 0.0
	frequency_penalty: float = 0.0
	repetition_penalty: float = 1.0
	metadata: vSurgeMetadata | None = None
	is_client_side_tokenization: bool = False

	@classmethod
	def from_sampling_params(cls, prompt: str, sampling_params: SamplingParams):
		return vSurgeRequest(
			prompt=prompt,
			max_tokens=sampling_params.max_tokens,
			top_p=sampling_params.top_p,
			top_k=sampling_params.top_k,
			min_p=sampling_params.min_p,
			temperature=sampling_params.temperature,
			presence_penalty=sampling_params.presence_penalty,
			frequency_penalty=sampling_params.frequency_penalty,
			repetition_penalty=sampling_params.repetition_penalty,
		)

	def __post_init__(self):
		"""Ensures metadata is initialized."""
		if self.metadata is None:
			self.metadata = vSurgeMetadata()
		self.is_client_side_tokenization = False


class _Exception:
	"""A class for propagating exceptions through a queue.

	By wrapping them with a custom private class we ensure that any type
	(including Exception) can be used as a V.
	"""

	def __init__(self, exception: Exception) -> None:
		self.exception = exception


class AsyncMultifuture(tp.Generic[V]):
	"""AsyncMultifuture is like concurrent.futures.Future but supports returning

	multiple results. It provides an unidirectional stream with buffering and
	exception propagation.

	Supports delivering results to an async Python event loop. Must be
	constructed inside of the event loop.
	"""

	def __init__(self) -> None:
		self._cancelled = threading.Event()
		self._done = threading.Event()
		self._loop = asyncio.get_running_loop()
		self._queue = asyncio.Queue[V | _Exception]()

	def cancel(self, unused: tp.Any = None) -> None:
		"""Cancels the asyncmultifuture."""
		# Needed for compatibility with grpc.aio.ServicerContext.add_done_callback.
		del unused
		self._cancelled.set()
		self.set_exception(futures.CancelledError())

	def cancelled(self) -> bool:
		"""Returns whether the asyncmultifuture has been cancelled."""
		return self._cancelled.is_set()

	def done(self) -> bool:
		"""AsyncMultifuture is done when it is finalized with close() or

		set_exception().
		"""
		return self._done.is_set()

	def set_exception(self, exception: Exception) -> None:
		"""Stores the given exception in the asyncmultifuture.

		The exception would be delivered after all previously added results are
		yielded. set_exception can be called multiple times, however subsequent
		calls will be ignored.

		Args:
		  exception: The exception to set.
		"""
		self._loop.call_soon_threadsafe(self._queue.put_nowait, _Exception(exception))
		self._loop.call_soon_threadsafe(self._done.set)

	def add_result(self, result: V) -> None:
		"""Adds the result to the asyncmultifuture.

		Caller must call .close() once all results are added.

		Args:
		  result: The result to add.
		"""
		self._loop.call_soon_threadsafe(self._queue.put_nowait, result)

	def close(self) -> None:
		"""Notifies the receiver that no more results would be added."""
		self.set_exception(StopAsyncIteration())

	def __aiter__(self) -> AsyncMultifuture:
		return self

	async def __anext__(self) -> V:
		"""Returns the next value."""
		value = await self._queue.get()
		if isinstance(value, _Exception):
			raise value.exception
		return value


@dataclasses.dataclass
class ActiveRequestMetadata:
	"""Inference request metadata."""

	start_time: tp.Optional[float] = None

	prefill_enqueue_time: tp.Optional[float] = None
	prefill_dequeue_time: tp.Optional[float] = None

	transfer_enqueue_time: tp.Optional[float] = None
	transfer_dequeue_time: tp.Optional[float] = None

	generate_enqueue_time: tp.Optional[float] = None
	generate_dequeue_time: tp.Optional[float] = None

	complete_time: tp.Optional[float] = None


@dataclasses.dataclass
class ActiveRequest:
	"""Current state of the driver."""

	max_tokens: int
	return_channel: AsyncMultifuture[list[ReturnSample]]
	top_p: float = 1.0
	top_k: int = 0
	min_p: float = 0.0
	temperature: float = 0.0
	presence_penalty: float = 0.0
	frequency_penalty: float = 0.0
	repetition_penalty: float = 1.0
	complete: tp.Optional[np.ndarray] = None
	prefill_result: tp.Any = None
	prefill_content: tp.Optional[str | list[int]] = None
	generate_timestep_added: tp.Optional[int] = None
	is_client_side_tokenization: tp.Optional[bool] = False
	# Metrics Tracking
	decode_start_time: float | None = None
	total_generated_tokens: int = 0
	metadata: ActiveRequestMetadata = dataclasses.field(
		default_factory=ActiveRequestMetadata
	)

	def enqueue_samples(self, generated_samples: list[ReturnSample]):
		"""Adds the generated sample(s) to return channel for current step.

		Args:
		  generated_samples: The generated sample(s) for current step.

		This should be called only from within the Drivers background thread.
		"""
		self.return_channel.add_result(generated_samples)


class JetThread(threading.Thread):
	"""Thread that kills the program if it fails.

	If a driver thread goes down, we can't operate.
	"""

	def run(self):
		"""Executes the thread's target function.

		If the target function raises any exception, this method catches it,
		prints the traceback, and forcefully kills the entire process using
		`os.kill` with `signal.SIGKILL`. This ensures that if a critical
		driver thread fails, the whole system stops, preventing potential
		inconsistent states or hangs.
		"""
		try:
			super().run()
		except Exception as e:
			print(f"Thread {self.name} encountered an error: {e}")
			traceback.print_exc()
			os.kill(os.getpid(), signal.SIGKILL)


class vDriver:
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

	def __init__(
		self,
		prefill_engines: tp.Optional[list[vEngine] | vEngine] = None,
		decode_engines: tp.Optional[list[vEngine] | vEngine] = None,
		interleaved_mode: bool = False,
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
		self._detokenize_backlogs = [queue.Queue(8) for _ in self._decode_engines]
		self._decode_slots = [
			queue.Queue(engine.max_concurrent_decodes) for engine in self._decode_engines
		]
		_ = [
			[self._decode_slots[idx].put(i) for i in range(engine.max_concurrent_decodes)]
			for idx, engine in enumerate(self._decode_engines)
		]

		self._prefill_threads = [
			JetThread(
				target=functools.partial(self._prefill_thread, idx),
				name=f"prefill-{idx}",
				daemon=True,
			)
			for idx in range(len(self._prefill_engines))
		]
		self._transfer_threads = [
			JetThread(
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
			JetThread(
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
			JetThread(
				target=functools.partial(
					self._detokenize_thread,
					idx,
				),
				name=f"detokenize-{idx}",
			)
			for idx in range(len(self._decode_engines))
		]
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

	def compile(self):
		"""Compiles the prefill engines."""
		self._compile_prefill()

	def _compile_prefill(self):
		"""Compiles the prefill engines for various sequence lengths."""
		for prefill_engine in self._prefill_engines:
			max_prefill_length = prefill_engine.max_prefill_length
			vals = prefill_engine.max_prefill_lengths[
				: prefill_engine.max_prefill_lengths.index(max_prefill_length)
			] + [max_prefill_length]
			for length in vals:
				padded_tokens = padded_valids = jnp.ones((1, length), "i4")
				logger.info(f"Compiling PrefillEngine seqlen={length}")
				prefill_engine.prefill(
					graphdef=prefill_engine.graphdef,
					graphstate=prefill_engine.graphstate,
					graphothers=prefill_engine.graphothers,
					tokens=padded_tokens,
					valids=padded_valids,
					max_length=prefill_engine.max_length,
					sampling_params=None,
					samples_per_slot=prefill_engine.samples_per_slot,
					rngs=prefill_engine.prng_key,
				)

	def stop(self):
		"""Stops the driver and all background threads."""
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
	def processor(self) -> ProcessingClassType:
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
		processor: ProcessingClassType,
		max_prefill_length: int,
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
			content = processor(content, return_tensors="np", return_attention_mask=True)
			tokens = jnp.array(content["input_ids"])
			valids = jnp.array(content["attention_mask"])
		else:
			tokens, valids = content
		sampling_params = SamplingParams(
			max_tokens=0,
			presence_penalty=request.presence_penalty,
			frequency_penalty=request.frequency_penalty,
			repetition_penalty=request.repetition_penalty,
			min_p=request.min_p,
			top_k=request.top_k,
			top_p=request.top_p,
			temperature=request.temperature,
		)
		return pad_tokens(
			tokens=tokens,
			valids=valids,
			pad_token_id=processor.pad_token_id,
			max_prefill_length=max_prefill_length,
		), sampling_params

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
			logger.info(
				f"Prefilling on prefill engine {idx} : "
				f"prefill queue size, {self._prefill_backlog.qsize()}",
			)
			(
				(
					padded_tokens,
					padded_valids,
					_,
				),
				sampling_params,
			) = self._process_prefill_content(
				request,
				processor,
				prefill_engine.max_prefill_length,
			)
			prefill_result, first_token = prefill_engine.prefill(
				graphdef=prefill_engine.graphdef,
				graphstate=prefill_engine.graphstate,
				graphothers=prefill_engine.graphothers,
				tokens=padded_tokens,
				valids=padded_valids,
				max_length=prefill_engine.max_length,
				sampling_params=sampling_params,
				samples_per_slot=prefill_engine.samples_per_slot,
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
			if (time.time() - time_of_last_print) > 1:
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

			decode_state, sampled_tokens = decode_engine.decode(
				graphdef=decode_engine.graphdef,
				graphstate=decode_engine.graphstate,
				graphothers=decode_engine.graphothers,
				state=decode_state,
				samples_per_slot=decode_engine.samples_per_slot,
				rngs=decode_engine.prng_key,
			)
			sampled_tokens.copy_to_host_async()
			my_detokenize_backlog.put((generate_timestep, sampled_tokens), block=True)
			generate_timestep += 1
			# TODO:Debug
			# logger.info(
			# 	"Decode engine %d step %d - slots free : %d / %d, took %.2fms",
			# 	idx,
			# 	generate_timestep,
			# 	my_slots_size,
			# 	max_concurrent_decodes,
			# 	(time.time() - time_of_last_decode) * 10**3,
			# )
			# time_of_last_decode = time.time()

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


class vSurge:
	"""Orchestrates the interaction between client requests and the vDriver."""

	def __init__(self, driver: vDriver, vsurge_name: str | None = None):
		"""Initializes the vSurge.

		Args:
		    driver: The vDriver instance that manages the underlying inference
		        engines and processing threads.
		"""
		self._driver = driver
		self._vsurge_name = vsurge_name or self._get_vsurge_name(
			driver._decode_engines[-1].model
		)

	def _get_vsurge_name(self, model) -> str:
		"""
		Generate a standardized vsurge name combining model type, size, and timestamp.

		Format: {model_type}-{size_in_B}B-{timestamp}
		Example: llama-7.00B-20240311
		"""
		model_type = self._get_model_type(model)
		model_size = self._calculate_model_size(model.graphstate)
		timestamp = datetime.datetime.now().strftime("%Y%m%d")

		return f"{model_type}-{model_size}B-{timestamp}"

	def _get_model_type(self, model) -> str:
		"""Get the model type, with fallback to 'unknown' if not found."""
		return getattr(model.config, "model_type", "unknown").lower()

	def _calculate_model_size(self, graphstate) -> str:
		"""
		Calculate model size in billions of parameters.
		Returns formatted string with 2 decimal places.
		"""
		try:
			num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
			size_in_billions = num_params / 1e9
			return f"{size_in_billions:.2f}"
		except Exception as e:
			logger.warning(f"Failed to calculate model size: {e}")
			return "unknown"

	def compile(self):
		self.driver.compile()

	@property
	def vsurge_name(self):
		return self._vsurge_name

	@property
	def driver(self):
		"""Provides access to the underlying vDriver instance."""
		return self._driver

	@property
	def processor(self) -> ProcessingClassType:
		"""Returns the processor/tokenizer associated with the underlying driver."""
		return self.driver.processor

	def stop(self):
		return self.driver.stop()

	@classmethod
	def create(
		cls,
		model: EasyDeLBaseModule,
		processor: ProcessingClassType,
		max_concurrent_decodes: int | None = None,
		max_prefill_lengths: int | None = None,
		max_prefill_length: int | None = None,
		max_length: int | None = None,
		seed: int = 894,
		vsurge_name: str | None = None,
	):
		"""Creates a new instance of vSurge with configured vDriver and vEngines.

		This class method provides a convenient way to instantiate the vSurge
		by setting up the necessary prefill and decode engines with the provided
		model, processor, and configuration parameters.

		Args:
		    model: The EasyDeLBaseModule instance representing the model.
		    processor: The tokenizer/processor instance.
		    max_concurrent_decodes: Maximum number of concurrent decode requests
		        the decode engine can handle.
		    max_prefill_lengths: A list of prefill lengths to compile for the
		        prefill engine.
		    max_prefill_length: The maximum prefill length for the prefill engine.
		    max_length: The maximum total sequence length for both engines.
		    seed: The random seed for reproducibility.
		    vsurge_name: An optional name for the vsurge.

		Returns:
		    A new instance of vSurge.
		"""
		return vSurge(
			driver=vDriver(
				prefill_engines=vEngine(
					model=model,
					processor=processor,
					max_concurrent_prefill=1,
					max_prefill_lengths=max_prefill_lengths,
					max_prefill_length=max_prefill_length,
					max_length=max_length,
					seed=seed,
				),
				decode_engines=vEngine(
					model=model,
					processor=processor,
					max_concurrent_decodes=max_concurrent_decodes,
					max_length=max_length,
					seed=seed,
				),
			),
			vsurge_name=vsurge_name,
		)

	def count_tokens(self, text_or_conversation: tp.Union[str, list]) -> int:
		"""Counts the number of tokens in a given string or conversation list.

		Uses the underlying driver's processor to tokenize the input and returns
		the count of tokens.

		Args:
		    text_or_conversation: Either a single string or a list of
		        message dictionaries (like OpenAI chat format).

		Returns:
		    The total number of tokens in the input.

		Raises:
		    ValueError: If the input type is invalid or tokenization fails.
		"""
		try:
			if isinstance(text_or_conversation, str):
				# Tokenize a single string
				return len(self.processor(text_or_conversation)["input_ids"])
			elif isinstance(text_or_conversation, list):
				if hasattr(self.processor, "apply_chat_template"):
					tokenized = self.processor.apply_chat_template(
						conversation=text_or_conversation,
						tokenize=True,
						add_generation_prompt=False,
					)
					return len(
						tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized
					)
				else:
					full_text = " ".join(
						[
							msg.get("content", "")
							for msg in text_or_conversation
							if isinstance(msg.get("content"), str)
						]
					)
					return len(self.processor(full_text)["input_ids"])
			else:
				raise ValueError(
					f"Unsupported input type for token counting: {type(text_or_conversation)}"
				)
		except Exception as e:
			logger.error(f"Error during token counting: {e}")
			# Re-raise or handle as appropriate for the API
			raise ValueError(f"Failed to count tokens: {e}") from e

	def process_client_side_tokenization_response(self, response: list[ReturnSample]):
		"""Processes responses when tokenization is handled client-side.

		In this case, the response items (ReturnSample) are typically yielded
		directly without further server-side processing like detokenization
		or buffering.

		Args:
		    response: A list of ReturnSample objects from a single generation step.

		Returns:
		    The input list of ReturnSample objects, unchanged.
		"""
		samples = []
		for sample in response:
			samples.append(sample)
		return samples

	def should_buffer_response(self, response: list[ReturnSample]) -> bool:
		"""Determines if a response needs buffering for server-side detokenization.

		Buffering is needed if any sample in the response ends with a byte token
		(e.g., "<0xAB>"), as this indicates an incomplete multi-byte character
		that requires subsequent tokens for proper decoding.

		Args:
		    response: A list of ReturnSample objects from a single generation step.

		Returns:
		    True if buffering is required, False otherwise.
		"""
		for item in response:
			# Check if the text list is not empty and the last item is a byte token
			if item.text and is_byte_token(item.text[-1]):
				return True
		return False  # Return False if no byte tokens found at the end

	def process_server_side_tokenization_response(
		self,
		response: list[ReturnSample],
		buffered_response_list: list[list[ReturnSample]],
	) -> list[ReturnSample]:
		"""Processes responses when tokenization/detokenization is server-side.

		Combines the text and token IDs from the current response and any
		buffered previous responses for each sample. It then uses the metrics
		(TPS, generated token count) from the *latest* response in the sequence
		for the final output.

		Args:
				  response: The list of ReturnSample objects from the current step.
				  buffered_response_list: A list containing lists of ReturnSample objects
				      from previous steps that were buffered.

		Returns:
				  A list of tuples, where each tuple represents a completed sample and
				  contains: (decoded_string, all_token_ids, latest_tps, latest_num_generated_tokens).
		"""
		current_response_with_flushed_buffer = list(zip(*buffered_response_list, response))
		current_response_with_flushed_buffer = tp.cast(
			list[list[ReturnSample]],
			current_response_with_flushed_buffer,
		)
		samples = []
		for sample_responses in current_response_with_flushed_buffer:
			text = []
			token_ids = []
			latest_response = sample_responses[-1]
			tps = latest_response.tokens_per_second
			num_gen_tokens = latest_response.num_generated_tokens
			for resp in sample_responses:
				text.extend(resp.text)
				token_ids.extend(resp.token_ids)
			samples.append(
				ReturnSample(
					text=text_tokens_to_string(text),
					token_ids=token_ids,
					tokens_per_second=tps,
					num_generated_tokens=num_gen_tokens,
				)
			)
		return samples

	async def complete(
		self,
		request: vSurgeRequest,
	) -> tp.AsyncGenerator[tp.List[ReturnSample]]:
		"""Initiates and streams the results of a text completion request.

		Creates an `ActiveRequest` using the plain prompt from the
		`vSurgeRequest`, places it on the driver's prefill queue,
		and then asynchronously iterates through the results provided by the
		`ActiveRequest`'s `return_channel`.

		It handles both client-side and server-side tokenization scenarios,
		buffering and processing results appropriately before yielding them.

		Args:
		    request: The vSurgeRequest containing the prompt and
		        generation parameters.

		Yields:
		    Processed generation results, similar to the `decode` method. The format
		    depends on the tokenization mode.

		Raises:
		    RuntimeError: If the prefill queue is full when trying to place the
		        request.
		"""
		return_channel = AsyncMultifuture()
		active_request = ActiveRequest(
			max_tokens=request.max_tokens,
			prefill_content=request.prompt,
			is_client_side_tokenization=request.is_client_side_tokenization,
			return_channel=return_channel,
			top_p=request.top_p,
			top_k=request.top_k,
			min_p=request.min_p,
			temperature=request.temperature,
			presence_penalty=request.presence_penalty,
			frequency_penalty=request.frequency_penalty,
			repetition_penalty=request.repetition_penalty,
			metadata=ActiveRequestMetadata(
				start_time=request.metadata.start_time if request.metadata else time.time(),
				prefill_enqueue_time=time.perf_counter(),
			),
		)
		try:
			self._driver.place_request_on_prefill_queue(active_request)
		except queue.Full as e:
			raise RuntimeError("Prefill queue is full") from e

		buffered_response_list = []
		async for response in active_request.return_channel:
			response = tp.cast(list[ReturnSample], response)
			if request.is_client_side_tokenization:
				yield self.process_client_side_tokenization_response(response)
			else:
				if self.should_buffer_response(response):
					buffered_response_list.append(response)
				else:
					yield self.process_server_side_tokenization_response(
						response,
						buffered_response_list,
					)
					buffered_response_list = []
		if not request.is_client_side_tokenization and buffered_response_list:
			last_real_response = buffered_response_list[-1]
			dummy_response = [
				ReturnSample(
					text=[],
					token_ids=[],
					tokens_per_second=s.tokens_per_second,
					num_generated_tokens=s.num_generated_tokens,
				)
				for s in last_real_response
			]
			yield self.process_server_side_tokenization_response(
				dummy_response,
				buffered_response_list,
			)
