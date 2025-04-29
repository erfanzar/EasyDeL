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
import queue
import time
import typing as tp

import jax

from easydel.inference.utilities import SamplingParams
from easydel.layers.caching.paged_attention import (
	HBMPageManager,
	PagedAttentionCache,
)
from easydel.utils.helpers import get_logger

from .engines import (
	oDriver,
	oEngine,
	vDriver,
	vEngine,
)
from .utils import (
	ActiveRequest,
	ActiveRequestMetadata,
	AsyncMultifuture,
	ReturnSample,
	is_byte_token,
	text_tokens_to_string,
)

if tp.TYPE_CHECKING:
	from easydel.infra.base_module import EasyDeLBaseModule
	from easydel.infra.utils import ProcessingClassType
else:
	ProcessingClassType = tp.Any
	EasyDeLBaseModule = tp.Any


logger = get_logger("vSurge")


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
	temperature: float = 0.7
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
		assert isinstance(self.prompt, str), "prompt should be a single string"


class vSurge:
	"""Orchestrates the interaction between client requests and the vDriver."""

	def __init__(
		self,
		driver: tp.Union[vDriver, oDriver],
		vsurge_name: str | None = None,
	):
		"""Initializes the vSurge.

		Args:
		    driver: The vDriver instance that manages the underlying inference
		        engines and processing threads.
		"""
		self._driver = driver
		self._vsurge_name = vsurge_name or driver.driver_name

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

	def start(self):
		return self.driver.start()

	def stop(self):
		return self.driver.stop()

	@classmethod
	def create_odriver(
		cls,
		model: EasyDeLBaseModule,
		processor: ProcessingClassType,
		storage: tp.Optional[PagedAttentionCache] = None,
		manager: tp.Optional[HBMPageManager] = None,
		page_size: int = 128,
		hbm_utilization: float = 0.6,
		max_concurrent_prefill: int | None = None,
		max_concurrent_decodes: int | None = None,
		prefill_lengths: int | None = None,
		max_prefill_length: int | None = None,
		max_length: int | None = None,
		seed: int = 894,
		vsurge_name: str | None = None,
	) -> vSurge:
		max_length = max_length or 8192
		max_concurrent_prefill = max_concurrent_prefill or jax.device_count()
		max_concurrent_decodes = max_concurrent_decodes or jax.device_count()
		metadata = model.create_paged_metadata(
			page_size=page_size,
			batch_size=max_concurrent_decodes,
			max_sequences=max_length,
			dtype=model.dtype,
			hbm_utilization=hbm_utilization,
		)
		if storage is None:
			storage = model.init_pages(metadata=metadata)
		if manager is None:
			manager = HBMPageManager(metadata=metadata)
		return vSurge(
			driver=oDriver(
				oEngine(
					model=model,
					processor=processor,
					storage=storage,
					manager=manager,
					max_concurrent_decodes=max_concurrent_decodes,
					max_concurrent_prefill=max_concurrent_prefill,
					prefill_lengths=prefill_lengths,
					max_prefill_length=max_prefill_length,
					max_length=max_length,
					batch_size=max_concurrent_decodes,
					seed=seed,
				)
			),
			vsurge_name=vsurge_name,
		)

	@classmethod
	def create_vdriver(
		cls,
		model: EasyDeLBaseModule,
		processor: ProcessingClassType,  
		max_concurrent_decodes: int | None = None,
		prefill_lengths: int | None = None,
		max_prefill_length: int | None = None,
		max_length: int | None = None,
		seed: int = 894,
		vsurge_name: str | None = None,
	) -> vSurge:
		"""Creates a new instance of vSurge with configured vDriver and vEngines.

		This class method provides a convenient way to instantiate the vSurge
		by setting up the necessary prefill and decode engines with the provided
		model, processor, and configuration parameters.

		Args:
		    model: The EasyDeLBaseModule instance representing the model.
		    processor: The tokenizer/processor instance.
		    max_concurrent_decodes: Maximum number of concurrent decode requests
		        the decode engine can handle.
		    prefill_lengths: A list of prefill lengths to compile for the
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
					prefill_lengths=prefill_lengths,
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
				return len(self.processor(text=text_or_conversation)["input_ids"])
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
					return len(self.processor(text=full_text)["input_ids"])
			else:
				raise ValueError(
					f"Unsupported input type for token counting: {type(text_or_conversation)}"
				)
		except Exception as e:
			logger.error(f"Error during token counting: {e}")
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

	async def generate(
		self,
		prompts: tp.Union[str, tp.Sequence[str]],
		sampling_params: tp.Optional[
			tp.Union[SamplingParams, tp.Sequence[SamplingParams]]
		] = None,
		stream: bool = False,
	) -> tp.Union[tp.List[ReturnSample], tp.AsyncGenerator[tp.List[ReturnSample]]]:
		"""Generates text completions concurrently for the given prompts.

		Args:
		  prompts: A single prompt string or a list of prompt strings.
		  sampling_params: A single SamplingParams object or a list of
		    SamplingParams objects. If None, default SamplingParams will be used.
		    If a single SamplingParams object is provided with multiple prompts,
		    it will be applied to all prompts. If a list is provided, it must
		    have the same length as the prompts list.
		  stream: If True, yields results (List[ReturnSample]) from *any* request
		    as they become available. The list corresponds to one generation step
		    from one request. If False, waits for all requests to complete and
		    returns a list containing one aggregated ReturnSample per prompt.

		Returns:
		  If stream is True: An async generator yielding lists of ReturnSample as
		    steps complete across concurrent requests.
		  If stream is False: A list of aggregated ReturnSample objects, one for
		    each input prompt, after all requests have finished.

		Raises:
		  ValueError: If the lengths of prompts and sampling_params lists mismatch.
		  RuntimeError: If the underlying driver's queue is full.
		"""

		if isinstance(prompts, str):
			prompts = [prompts]
			if sampling_params is not None and not isinstance(
				sampling_params, SamplingParams
			):
				raise ValueError(
					"If prompts is a single string, sampling_params must be a single SamplingParams object or None."
				)
			sampling_params = [sampling_params if sampling_params else SamplingParams()]
		elif isinstance(prompts, tp.Sequence):
			if sampling_params is None:
				sampling_params = [SamplingParams()] * len(prompts)
			elif isinstance(sampling_params, SamplingParams):
				sampling_params = [sampling_params] * len(prompts)
			elif isinstance(sampling_params, tp.Sequence):
				if len(prompts) != len(sampling_params):
					raise ValueError("Lengths of prompts and sampling_params lists must match.")
			else:
				raise ValueError(
					"sampling_params must be a SamplingParams object, a list of SamplingParams objects, or None."
				)
		else:
			raise ValueError("prompts must be a string or a sequence of strings.")

		if not prompts:
			if stream:

				async def empty_generator():
					if False:
						yield []

				return empty_generator()
			else:
				return []

		requests = [
			vSurgeRequest.from_sampling_params(prompt=p, sampling_params=sp)
			for p, sp in zip(prompts, sampling_params)
		]

		if stream:
			return self._generate_stream(requests)
		else:
			return await self._generate_batch(requests)

	async def _generate_stream(
		self, requests: tp.List[vSurgeRequest]
	) -> tp.AsyncGenerator[tp.List[ReturnSample]]:
		"""Helper for concurrent streaming generation."""
		q = asyncio.Queue()
		tasks = set()
		results_map: tp.Dict[asyncio.Task, tp.List[ReturnSample]] = {}
		_SENTINEL = object()

		async def _run_completion(request: vSurgeRequest):
			"""Runs self.complete and puts results (or sentinel/exception) in queue."""
			try:
				async for result_step in self.complete(request):
					await q.put(result_step)
			except Exception as e:
				await q.put(e)
			finally:
				await q.put(_SENTINEL)

		for req in requests:
			task = asyncio.create_task(_run_completion(req))
			tasks.add(task)
			results_map[task] = []
		finished_tasks = 0
		while finished_tasks < len(requests):
			item = await q.get()
			if item is _SENTINEL:
				finished_tasks += 1
			elif isinstance(item, Exception):
				for task in tasks:
					if not task.done():
						task.cancel()
				await asyncio.gather(*tasks, return_exceptions=True)
				raise item
			else:
				yield item
			q.task_done()
		await asyncio.gather(*tasks, return_exceptions=True)

	async def _generate_batch(
		self,
		requests: tp.List[vSurgeRequest],
	) -> tp.List[ReturnSample]:
		"""Helper for concurrent batch generation."""

		async def _collect_all_steps(request: vSurgeRequest) -> list[list[ReturnSample]]:
			"""Consumes the complete generator and returns all steps."""
			all_steps = []
			async for step_result in self.complete(request):
				all_steps.append(step_result)
			return all_steps

		tasks = [asyncio.create_task(_collect_all_steps(req)) for req in requests]
		try:
			results_per_request: list[list[list[ReturnSample]]] = await asyncio.gather(*tasks)
		except Exception as e:
			for task in tasks:
				if not task.done():
					task.cancel()
			await asyncio.gather(*tasks, return_exceptions=True)
			raise e

		final_results: tp.List[ReturnSample] = []
		for request_steps in results_per_request:
			if not request_steps:
				final_results.append(
					ReturnSample(
						text="", token_ids=[], tokens_per_second=0.0, num_generated_tokens=0
					)
				)
				continue

			aggregated_text = ""
			aggregated_token_ids = []
			total_tps = 0.0
			final_num_generated_tokens = 0
			num_steps = 0

			for step_result in request_steps:
				if step_result:
					sample = step_result[0]
					aggregated_text += sample.text
					aggregated_token_ids.extend(sample.token_ids)
					total_tps += sample.tokens_per_second
					final_num_generated_tokens = sample.num_generated_tokens
					num_steps += 1

			average_tps = (total_tps / num_steps) if num_steps > 0 else 0.0

			final_results.append(
				ReturnSample(
					text=aggregated_text,
					token_ids=aggregated_token_ids,
					tokens_per_second=average_tps,
					num_generated_tokens=final_num_generated_tokens,
				)
			)

		return final_results
