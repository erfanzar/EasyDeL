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

import numpy as np

from easydel.layers.caching.paged_attention import (
	ActiveSequenceBatch,
	GenerationStepTask,
	InitialSequenceRequest,
	ModelOutputSummary,
	NextIterationPlan,
	SamplingParams,
)
from easydel.utils.helpers import capture_time, get_logger

from ...utils import ActiveRequest, ReturnSample, SafeThread
from .._abstract_driver import AbstractDriver, ProcessingClassType
from .engine import oEngine

if tp.TYPE_CHECKING:
	from easydel.infra.utils import ProcessingClassType
else:
	ProcessingClassType = tp.Any


logger = get_logger("vSurge-oDriver")


class oDriver(AbstractDriver):
	"""
	oDriver is responsible for managing the inference process for the oEngine.
	It handles request submission, input preparation, inference execution,
	and processing of model outputs. It utilizes background threads for
	concurrent processing of different stages of the inference pipeline.
	"""

	_active_requests: list[queue.Queue[tuple[int, ActiveRequest]]] = []

	def __init__(self, engine: oEngine):
		"""
		Initializes the oDriver with a given oEngine.

		Args:
		  engine (oEngine): The inference engine to be driven.
		"""
		self.engine = engine
		metadata = engine.manager.metadata
		self.metadata = metadata
		# Semaphore to limit the number of concurrent requests being processed
		self._max_locks = metadata.batch_size * 3 // 2
		self._max_allowed_requests = threading.Semaphore(self._max_locks)
		self._prepare_backlog: queue.Queue[ActiveRequest] = queue.Queue()
		self._process_backlog: queue.Queue[ModelOutputSummary] = queue.Queue()
		self._active_state = None
		self._active_requests_map: tp.Dict[str, ActiveRequest] = {}
		self._decode_state = engine.init_decode_state()
		self._process_summery_thread = SafeThread(
			target=self._process_summery,
			name="_process_summery",
		)
		# Thread for executing model inference
		self._execution_loop_thread = SafeThread(
			target=self._execution_loop,
			name="_execution_loop",
		)
		# Thread for preparing inputs for inference
		self._prepare_inputs_thread = SafeThread(
			target=self._prepare_inputs,
			name="_prepare_inputs",
		)
		# Flag indicating if the driver is live and running
		self.live = False
		self.start()

	@property
	def driver_name(self):
		"""
		Returns the name of the driver, derived from the engine's model name.
		"""
		return self._get_model_name(self.engine.model)

	@property
	def num_used_slots(self):
		return self._max_locks - self._max_allowed_requests._value

	def compile(self):
		"""
		Compiles the underlying engines.

		This method is intended to perform any necessary compilation steps for the
		inference engines. Currently, it's a placeholder.
		"""
		try:
			self._compile_decode()
		except Exception:
			traceback.print_exc()
			self.stop()
			exit(1)

	def _compile_decode(self):
		"""
		Compiles the underlying engines.

		This method is intended to perform any necessary compilation steps for the
		inference engines. Currently, it's a placeholder.
		"""
		try:
			logger.info("Compiling Decode Engine")
			with capture_time() as cap:
				self.engine.forward(
					self.engine.graphstate,
					self.engine.graphothers,
					ActiveSequenceBatch.create(self.engine.metadata, self.engine.model.mesh),
					NextIterationPlan(
						schedule_prefill=False,
						prefill_request=None,
						prefill_pages_update=None,
						schedule_decodes=True,
						new_decodes_requests=[],
						decodes_state_page_updates=[],
					),
				)

			logger.info(f"Decode Engine Compiled in {cap()}")

		except Exception:
			traceback.print_exc()
			self.stop()
			exit(1)

	def start(self):
		"""
		Starts the driver and its background processing threads.

		Threads for input preparation, inference, and summary processing are started
		if the driver is not already live.
		"""
		if not self.live:
			self._process_summery_thread.start()
			self._execution_loop_thread.start()
			self._prepare_inputs_thread.start()
			self.live = True

	def stop(self):
		"""
		Stops the driver and all background threads gracefully.

		Signals the background threads to exit by putting None into their respective
		queues and then waits for them to join.
		"""
		if self.live:
			self.live = False
			# Signal threads to exit
			self._prepare_backlog.put(None)
			self._process_backlog.put(None)
			self.engine.scheduler.enqueue_prefill_request(None)
			self.engine.scheduler.enqueue_decodes_request(None)
			# Wait for threads to finish
			self._process_summery_thread.join()
			self._execution_loop_thread.join()
			self._prepare_inputs_thread.join()

	def submit_request(self, request: tp.Any):
		"""
		Submits a new request to the driver's processing pipeline.

		The request is placed on the prefill queue for initial processing.

		Args:
		  request (tp.Any): The request object to submit. Must be of type ActiveRequest.

		Raises:
		  TypeError: If the submitted request is not an instance of ActiveRequest.
		"""
		if not isinstance(request, ActiveRequest):
			raise TypeError("Request must be of type ActiveRequest")
		self.place_request_on_prefill_queue(request)

	def get_total_concurrent_requests(self) -> int:
		"""
		Gets the total number of concurrent requests the driver can handle.

		This is determined by the maximum number of concurrent decode operations
		supported by the engine.

		Returns:
		  int: The maximum number of concurrent requests.
		"""

		return self.engine.max_concurrent_decodes

	def place_request_on_prefill_queue(self, request: ActiveRequest):
		"""
		Places a new request onto the prefill queue for processing.

		This method is used internally to add requests that require prefilling
		and subsequent generation.

		Args:
		  request (ActiveRequest): The active request to place on the queue.
		"""
		self._prepare_backlog.put(request, block=False)

	@property
	def processor(self) -> ProcessingClassType:  # type:ignore
		"""
		Returns the processor/tokenizer associated with the engines.

		Assumes all engines (prefill and decode) use the same processor.
		Raises an error if no engines are configured.
		"""
		return self.engine.processor

	def _get_chunksize(self, length):
		return 512
		# prefill_chunk_sizes = [128, 256, 512]
		# for size in prefill_chunk_sizes:
		# 	if length <= size:
		# 		return size
		# return prefill_chunk_sizes[-1]

	def _prepare_inputs(self):
		"""
		Background thread method for preparing inputs for inference.

		It continuously retrieves requests from the prepare backlog,
		creates an InitialSequenceRequest, and enqueues it for prefill processing
		by the engine's scheduler. It also acquires a semaphore to limit
		the number of active requests.
		"""
		while True:
			request = self._prepare_backlog.get(block=True)
			if request is None:
				break
			logger.info(
				f"PrepareInputs: Received request {request.id}. Prepare backlog size: {self._prepare_backlog.qsize()}"
			)
			self._active_requests_map[request.id] = request
			self._max_allowed_requests.acquire()

			logger.info(
				f"PrepareInputs: Creating prefill request {request.id} for processing."
			)
			prompt_token_ids = (
				self.engine.processor.encode(
					request.prefill_content,
					return_tensors="np",
				)
				.ravel()
				.tolist()
			)
			initial_request = InitialSequenceRequest.create(
				id=request.id,
				mesh=self.engine.model.mesh,
				metadata=self.metadata,
				chunk_size=self._get_chunksize(len(prompt_token_ids)),
				prompt_token_ids=prompt_token_ids,
				max_prefill_length=self.engine.max_prefill_length,
				prefill_lengths=self.engine.prefill_lengths,
				sampling_params=SamplingParams(
					top_p=request.top_p,
					max_tokens=request.max_tokens,
					temperature=request.temperature,
				),
			)

			logger.info(
				f"PrepareInputs: Enqueuing prefill request {initial_request.id} for processing."
			)

			self.engine.scheduler.enqueue_prefill_request(initial_request)

	def _execution_loop(self):
		"""
		Background thread method for executing model inference.

		It continuously creates an iteration plan using the engine's scheduler,
		executes the engine's forward pass based on the plan, and processes
		the resulting output summary. It handles both prefill and decode steps
		and updates the internal state accordingly.
		"""
		while True:
			# Create an iteration plan based on the current active and decode states
			iteration_plan = self.engine.scheduler.create_plan(
				self._active_state,
				self._decode_state,
			)
			if iteration_plan is None:
				# Exit loop if no iteration plan can be created (signaling shutdown or no active requests)
				logger.info("No iteration plan created, exiting inference loop.")
				return
			logger.debug(
				f"Created iteration plan. Prefill: {iteration_plan.schedule_prefill}, "
				f"Decode: {iteration_plan.schedule_decodes}"
			)
			logger.debug("Executing engine.forward for plan.")
			# Execute the forward pass of the engine based on the iteration plan
			with capture_time() as took:
				output = self.engine.forward(
					self.engine.graphstate,
					self.engine.graphothers,
					self._decode_state,
					iteration_plan,
				)
			_exec = took() * 10**3
			logger.info(
				f"engine - used slots : {self.num_used_slots} / {self._max_locks}, "
				f"execution time {_exec:.2f}ms "
			)
			logger.debug("engine.forward call completed.")
			# Create a ModelOutputSummary from the engine's output
			summary = ModelOutputSummary.from_output(output)

			if iteration_plan.schedule_prefill:
				prefill = iteration_plan.prefill_request
				prefill.chunk_idx += 1
				start_idx = prefill.chunk_idx * prefill.chunk_size
				prefill_length = len(prefill.prompt_token_ids)
				# If there are more chunks to prefill, update the active state
				if start_idx < prefill_length:
					self._active_state = prefill
				else:
					# If prefill is complete, clear the active state and enqueue for decode
					self._active_state = None
					summary.prefill_request_id = iteration_plan.prefill_request.id
					decodes_request = GenerationStepTask(
						id=prefill.id,
						slot=-1,  # Slot will be assigned by the scheduler
						position=prefill_length,
						page_indices=prefill.page_indices,
						prefill_token_id=output.prefill_token_id,
						sampling_params=prefill.sampling_params,
					)
					self.engine.scheduler.enqueue_decodes_request(decodes_request)

			# Handle decode scheduling and state updates
			if iteration_plan.schedule_decodes:
				self._decode_state.token_ids = output.decodes_token_ids
				self._decode_state.positions = output.decodes_next_position
				self._decode_state.sampling_params = output.next_sampling_params

				with self._decode_state.context_lock:
					for (
						slot,
						processed_request,
					) in self._decode_state.active_slot_requests_map.items():
						processed_request.position += 1
						summary.decodes_active_slots.append(slot)
						summary.decodes_active_request_ids.append(processed_request.id)
			logger.debug(
				f"Placing summary onto process backlog. Backlog size: {self._process_backlog.qsize()}"
			)
			# Put the summary onto the process backlog for further handling
			self._process_backlog.put(summary)

	def _process_summery(self):
		"""
		Background thread method for processing model output summaries.

		It continuously retrieves summaries from the process backlog,
		handles completed prefill and decode requests, frees memory pages
		for completed requests, calculates and updates Tokens Per Second (TPS),
		and enqueues generated samples for the respective requests.
		"""
		while True:
			summary = self._process_backlog.get(block=True)
			if summary is None:
				logger.info("Received None, exiting summary processing loop.")
				return
			logger.debug(
				f"Received summary. Process backlog size: {self._process_backlog.qsize()}"
			)
			assert isinstance(summary, ModelOutputSummary)

			summary.prefill_token_id = np.asarray(summary.prefill_token_id).item()
			summary.prefill_complete = np.asarray(summary.prefill_complete).item()

			summary.decodes_token_ids = np.asarray(summary.decodes_token_ids).tolist()
			summary.decodes_completes = np.asarray(summary.decodes_completes).tolist()

			# Handle completion of decode requests
			if len(summary.decodes_active_request_ids) > 0:
				with self._decode_state.context_lock:
					logger.debug(
						f"[Decode Cleanup]: Checking {len(self._decode_state.active_slot_requests_map)} active slots for completion."
					)
					slots_to_free = []
					# Iterate through active slots to check for completed requests
					for slot in list(self._decode_state.active_slot_requests_map.keys()):
						if (
							slot < len(summary.decodes_completes) and summary.decodes_completes[slot]
						):
							pages_to_free = self._decode_state.active_slot_requests_map[
								slot
							].page_indices
							logger.debug(
								f"[Decode Cleanup]: Request in slot {slot} completed. Freeing slot and {len(pages_to_free)} pages."
							)
							self._decode_state.available_slots.put(slot, block=True)
							self.engine.manager.free_hbm_pages(pages_to_free)

							slots_to_free.append(slot)

					for slot in slots_to_free:
						if slot in self._decode_state.active_slot_requests_map:
							del self._decode_state.active_slot_requests_map[slot]

			# Handle completion of prefill requests
			if summary.prefill_request_id:
				request = self._active_requests_map[summary.prefill_request_id]
				request.decode_start_time = None
				request.total_generated_tokens = 0
				logger.info(f"Processed prefill token for request {request.id}.")
				request.enqueue_samples(
					[
						ReturnSample(
							text=[
								self.processor.decode(
									summary.prefill_token_id,
									skip_special_tokens=True,
								)
							],
							token_ids=[summary.prefill_token_id],
							tokens_per_second=0.0,  # TPS is not applicable for the single prefill token
							num_generated_tokens=1,  # Count the prefill token
						)
					],
				)
				# If prefill is complete, mark the request as complete, close its channel,
				# remove it from the active requests map, and release the semaphore.
				if summary.prefill_complete:
					logger.info(
						f"[Prefill]: Prefill complete for request {request.id}. Closing channel and releasing semaphore."
					)
					request.complete = True
					request.return_channel.close()
					del self._active_requests_map[summary.prefill_request_id]
					self._max_allowed_requests.release()

			# Process decode tokens for active requests
			for slot, request_id in zip(
				summary.decodes_active_slots,
				summary.decodes_active_request_ids,
			):
				# Skip if the request is no longer in the active requests map (e.g., completed in a previous summary)
				if request_id not in self._active_requests_map:
					continue
				# Retrieve the active request object
				request = self._active_requests_map[request_id]

				# Initialize timing/count if not done (e.g., if prefill stage was skipped or missed)
				if not hasattr(request, "decode_start_time"):
					request.decode_start_time = None
				if not hasattr(request, "total_generated_tokens"):
					request.total_generated_tokens = 0

				# Start timer on the first actual decode step for this request
				if request.decode_start_time is None:
					request.decode_start_time = time.perf_counter()
					logger.info(f"Started decode timing for request {request.id} at slot {slot}.")

				request.total_generated_tokens += 1
				elapsed_time = time.perf_counter() - request.decode_start_time
				tokens_per_second = (
					request.total_generated_tokens / elapsed_time if elapsed_time > 1e-6 else 0.0
				)

				request.enqueue_samples(
					[
						ReturnSample(
							text=[
								self.processor.decode(
									summary.decodes_token_ids[slot],
									skip_special_tokens=True,
								)
							],
							token_ids=[summary.decodes_token_ids[slot]],
							tokens_per_second=tokens_per_second,
							num_generated_tokens=request.total_generated_tokens,
						)
					],
				)

				# Check if the decode for this request is complete
				if slot < len(summary.decodes_completes) and summary.decodes_completes[slot]:
					logger.info(
						f"Decode complete for request {request.id} in slot {slot}. Closing channel and releasing semaphore."
					)
					# Mark the request as complete, close its channel, and release the semaphore
					request.complete = True
					request.return_channel.close()
					self._max_allowed_requests.release()
					# Remove the completed request from the active requests map
					if request_id in self._active_requests_map:
						del self._active_requests_map[request_id]
