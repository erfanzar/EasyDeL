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

"""
This module defines the vSurge system, a high-throughput inference engine
for EasyDeL models. It orchestrates text generation requests, managing
the underlying driver (vDriver or oDriver) and processing responses.
"""

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
	"""
	Tracks timing information for requests processed by the vSurge.

	Attributes:
	    start_time (float): The Unix timestamp (seconds) when the request processing started.
	"""

	def __init__(self):
		"""
		Initializes the metadata, capturing the current time as the start time.
		"""
		self.start_time = time.time()


@dataclasses.dataclass
class vSurgeRequest:
	"""
	Represents a request specifically for text completion.

	This dataclass encapsulates all parameters necessary for a text generation
	request within the vSurge system.

	Attributes:
	    prompt (str): The input prompt for text completion.
	    max_tokens (int): The maximum number of tokens to generate.
	    top_p (float): The nucleus sampling probability. Only tokens comprising the top_p
	        cumulative probability are considered. Defaults to 1.0.
	    top_k (int): The number of highest probability vocabulary tokens to keep for
	        top-k-filtering. Defaults to 0 (no top-k filtering).
	    min_p (float): The minimum probability for a token to be considered. Defaults to 0.0.
	    stop (tp.Optional[tp.Union[str, tp.List[str]]]): A string or list of strings
	        that, if generated, will cause the generation to stop. Defaults to None.
	    temperature (float): The sampling temperature. Higher values make the output more
	        random, lower values make it more deterministic. Defaults to 0.7.
	    presence_penalty (float): Penalty applied to tokens based on their presence in the
	        generated text so far. Discourages repetition. Defaults to 0.0.
	    frequency_penalty (float): Penalty applied to tokens based on their frequency in the
	        generated text so far. Discourages frequent tokens. Defaults to 0.0.
	    repetition_penalty (float): Penalty applied to repeated tokens. A value > 1.0
	        discourages repetition. Defaults to 1.0.
	    metadata (vSurgeMetadata | None): Metadata associated with the request, such as
	        start time. Automatically initialized if None.
	    is_client_side_tokenization (bool): If True, indicates that the prompt is already
	        tokenized and the client expects token IDs as output. Defaults to False.
	"""

	prompt: str
	max_tokens: int

	top_p: float = 1.0
	top_k: int = 0
	min_p: float = 0.0

	stop: tp.Optional[tp.Union[str, tp.List[str]]] = None

	temperature: float = 0.7
	presence_penalty: float = 0.0
	frequency_penalty: float = 0.0
	repetition_penalty: float = 1.0
	metadata: vSurgeMetadata | None = None
	is_client_side_tokenization: bool = False

	@classmethod
	def from_sampling_params(cls, prompt: str, sampling_params: SamplingParams):
		"""
		Creates a vSurgeRequest instance from a prompt and SamplingParams.

		Args:
		    prompt (str): The input prompt string.
		    sampling_params (SamplingParams): An object containing sampling parameters.

		Returns:
		    vSurgeRequest: A new vSurgeRequest instance initialized with the
		        provided prompt and sampling parameters.
		"""
		return vSurgeRequest(
			prompt=prompt,
			max_tokens=sampling_params.max_tokens,
			top_p=sampling_params.top_p,
			top_k=sampling_params.top_k,
			min_p=sampling_params.min_p,
			stop=sampling_params.stop,
			temperature=sampling_params.temperature,
			presence_penalty=sampling_params.presence_penalty,
			frequency_penalty=sampling_params.frequency_penalty,
			repetition_penalty=sampling_params.repetition_penalty,
		)

	def __post_init__(self):
		"""
		Ensures metadata is initialized and validates the prompt type.
		`is_client_side_tokenization` is also explicitly set to False by default here.
		"""
		if self.metadata is None:
			self.metadata = vSurgeMetadata()
		self.is_client_side_tokenization = False
		assert isinstance(self.prompt, str), "prompt should be a single string"


class vSurge:
	"""
	Orchestrates high-throughput text generation by interacting with a vDriver or oDriver.

	The vSurge class acts as the main interface for submitting text generation
	requests. It manages the underlying inference driver, handles request queuing,
	and processes responses, including tokenization and detokenization if needed.

	Attributes:
	    _driver (tp.Union[vDriver, oDriver]): The underlying inference driver instance
	        (either vDriver for standard attention or oDriver for paged attention).
	    _vsurge_name (str): The name of this vSurge instance, defaulting to the
	        driver's name.
	"""

	def __init__(
		self,
		driver: tp.Union[vDriver, oDriver],
		vsurge_name: str | None = None,
	):
		"""
		Initializes the vSurge instance.

		Args:
		    driver (tp.Union[vDriver, oDriver]): The underlying driver instance
		        (vDriver or oDriver) that will handle the actual inference.
		    vsurge_name (str | None): An optional name for this vSurge instance.
		        If None, it defaults to the name of the provided driver.
		"""
		self._driver = driver
		self._vsurge_name = vsurge_name or driver.driver_name

	def compile(self):
		"""
		Compiles the underlying driver.

		This typically involves JIT compilation of the model's forward pass for
		optimized execution.
		"""
		self.driver.compile()

	@property
	def vsurge_name(self) -> str:
		"""
		Returns the name of the vSurge instance.

		Returns:
		    str: The name of this vSurge instance.
		"""
		return self._vsurge_name

	@property
	def driver(self) -> tp.Union[vDriver, oDriver]:
		"""
		Provides access to the underlying driver instance.

		Returns:
		    tp.Union[vDriver, oDriver]: The vDriver or oDriver instance used by
		        this vSurge.
		"""
		return self._driver

	@property
	def processor(self) -> ProcessingClassType:
		"""
		Returns the processor/tokenizer associated with the underlying driver.

		The processor is used for tokenizing prompts and detokenizing generated
		token IDs.

		Returns:
		    ProcessingClassType: The processor (e.g., a Hugging Face tokenizer)
		        instance.
		"""
		return self.driver.processor

	def start(self):
		"""
		Starts the underlying driver.

		This initializes the driver's processing loops and makes it ready to
		accept inference requests.
		"""
		return self.driver.start()

	def stop(self):
		"""
		Stops the underlying driver.

		This gracefully shuts down the driver's processing loops.
		"""
		return self.driver.stop()

	def pause(self):
		"""
		Pauses the underlying driver.

		This temporarily halts the processing of new requests by the driver.
		"""
		return self.driver.pause()

	def resume(self):
		"""
		Resumes the underlying driver after it has been paused.

		This allows the driver to continue processing requests.
		"""
		return self.driver.resume()

	def replace_graphstate(self, state):
		"""
		Replaces the graph state of the underlying driver.

		This is an advanced feature, typically used for dynamic model updates or
		state management in complex scenarios.

		Args:
		    state: The new graph state to be applied to the driver.
		"""
		return self.driver.replace_graphstate(state=state)

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
		verbose: bool = True,
	) -> "vSurge":
		"""
		Creates a new instance of vSurge with an oDriver (PagedAttention).

		This factory method simplifies the creation of a vSurge instance that
		utilizes paged attention for memory-efficient inference.

		Args:
		    model (EasyDeLBaseModule): The EasyDeL model instance.
		    processor (ProcessingClassType): The processor/tokenizer instance.
		    storage (tp.Optional[PagedAttentionCache]): The storage for paged
		        attention cache. If None, it will be initialized.
		    manager (tp.Optional[HBMPageManager]): The HBM page manager. If None,
		        it will be initialized.
		    page_size (int): The size of each page in the paged attention cache.
		        Defaults to 128.
		    hbm_utilization (float): The target HBM utilization for the paged
		        attention cache. Defaults to 0.6 (60%).
		    max_concurrent_prefill (int | None): The maximum number of concurrent
		        prefill operations. Defaults to the number of available JAX devices.
		    max_concurrent_decodes (int | None): The maximum number of concurrent
		        decode operations. Defaults to the number of available JAX devices.
		    prefill_lengths (int | None): Specific prefill lengths to optimize for.
		    max_prefill_length (int | None): The maximum length for prefill operations.
		    max_length (int | None): The maximum total sequence length (prompt + generation).
		        Defaults to 8192.
		    seed (int): The random seed for reproducibility. Defaults to 894.
		    vsurge_name (str | None): An optional name for the vSurge instance.
		    verbose (bool): Whether to enable verbose logging for the driver.
		        Defaults to True.

		Returns:
		    vSurge: A new vSurge instance configured with an oDriver.
		"""
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
				engine=oEngine(
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
				),
				verbose=verbose,
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
		vsurge_name: str | None = None,
		verbose: bool = True,
		seed: int = 894,
	) -> "vSurge":
		"""
		Creates a new instance of vSurge with a vDriver (standard attention).

		This factory method simplifies the creation of a vSurge instance that
		utilizes standard attention mechanisms.

		Args:
		    model (EasyDeLBaseModule): The EasyDeL model instance.
		    processor (ProcessingClassType): The processor/tokenizer instance.
		    max_concurrent_decodes (int | None): The maximum number of concurrent
		        decode operations.
		    prefill_lengths (int | None): Specific prefill lengths to optimize for.
		    max_prefill_length (int | None): The maximum length for prefill operations.
		    max_length (int | None): The maximum total sequence length (prompt + generation).
		    vsurge_name (str | None): An optional name for the vSurge instance.
		    verbose (bool): Whether to enable verbose logging for the driver.
		        Defaults to True.
		    seed (int): The random seed for reproducibility. Defaults to 894.

		Returns:
		    vSurge: A new vSurge instance configured with a vDriver.
		"""
		return vSurge(
			driver=vDriver(
				prefill_engines=vEngine(
					model=model,
					processor=processor,
					max_concurrent_prefill=1,  # vDriver typically handles prefill serially per engine
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
				verbose=verbose,
			),
			vsurge_name=vsurge_name,
		)

	def count_tokens(self, text_or_conversation: tp.Union[str, list]) -> int:
		"""
		Counts the number of tokens in a given string or conversation list.

		Uses the underlying driver's processor to tokenize the input. If the input
		is a list (assumed to be a conversation in OpenAI chat format), it attempts
		to apply the chat template if available, otherwise concatenates content fields.

		Args:
		    text_or_conversation (tp.Union[str, list]): Either a single string or a
		        list of message dictionaries (e.g., `[{"role": "user", "content": "Hello"}]`).

		Returns:
		    int: The total number of tokens in the input.

		Raises:
		    ValueError: If the input type is unsupported or if tokenization fails.
		"""
		try:
			if isinstance(text_or_conversation, str):
				return len(self.processor(text=text_or_conversation)["input_ids"])
			elif isinstance(text_or_conversation, list):
				if hasattr(self.processor, "apply_chat_template"):
					# Ensure add_generation_prompt=False to count only input tokens
					tokenized = self.processor.apply_chat_template(
						conversation=text_or_conversation,
						tokenize=True,
						add_generation_prompt=False,  # Important for accurate input token count
					)
					return len(
						tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized
					)
				else:
					# Fallback for tokenizers without apply_chat_template
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
		"""
		Processes responses when tokenization is handled client-side.

		In this mode, the `ReturnSample` objects, which already contain token IDs
		and potentially raw text segments, are typically passed through directly.
		The client is responsible for any further detokenization or assembly.

		Args:
		    response (list[ReturnSample]): A list of `ReturnSample` objects from a
		        single generation step, where each sample corresponds to a request
		        in a batch.

		Returns:
		    list[ReturnSample]: The input list of `ReturnSample` objects, unchanged.
		"""
		samples = []
		for sample in response:
			samples.append(sample)
		return samples

	def should_buffer_response(self, response: list[ReturnSample]) -> bool:
		"""
		Determines if a response needs buffering for server-side detokenization.

		Buffering is necessary if any `ReturnSample` in the response ends with a
		byte token (e.g., "<0xAB>"). This indicates an incomplete multi-byte
		UTF-8 character that requires subsequent tokens for proper decoding.

		Args:
		    response (list[ReturnSample]): A list of `ReturnSample` objects from a
		        single generation step.

		Returns:
		    bool: True if buffering is required (a byte token is found at the end
		        of any sample's text list), False otherwise.
		"""
		for item in response:
			if item.text and is_byte_token(item.text[-1]):
				return True
		return False

	def process_server_side_tokenization_response(
		self,
		response: list[ReturnSample],
		buffered_response_list: list[list[ReturnSample]],
	) -> list[ReturnSample]:
		"""
		Processes responses when tokenization/detokenization is server-side.

		This method combines text segments and token IDs from the current response
		and any previously buffered responses for each sample in a batch. It then
		detokenizes the combined text segments. Metrics like tokens per second (TPS)
		and the number of generated tokens are taken from the latest `ReturnSample`
		in the sequence for each request.

		Args:
		    response (list[ReturnSample]): The list of `ReturnSample` objects from
		        the current generation step.
		    buffered_response_list (list[list[ReturnSample]]): A list where each
		        inner list contains `ReturnSample` objects from previous, buffered
		        steps for corresponding requests.

		Returns:
		    list[ReturnSample]: A list of `ReturnSample` objects, where each object
		        contains the fully detokenized string for the current step (including
		        buffered parts), all accumulated token IDs, and the latest performance
		        metrics.
		"""
		# Transpose: group responses by original request
		# buffered_response_list: [[req1_step1, req2_step1], [req1_step2, req2_step2]]
		# response: [req1_step3, req2_step3]
		# current_response_with_flushed_buffer will be:
		# [(req1_step1, req1_step2, req1_step3), (req2_step1, req2_step2, req2_step3)]
		current_response_with_flushed_buffer = list(zip(*buffered_response_list, response))
		current_response_with_flushed_buffer = tp.cast(
			list[list[ReturnSample]],  # Each inner list is for one original request
			current_response_with_flushed_buffer,
		)
		samples = []
		for (
			sample_responses
		) in current_response_with_flushed_buffer:  # Iterates per original request
			text_segments_for_detok = []
			all_token_ids = []

			latest_response_for_this_sample = sample_responses[-1]
			tps = latest_response_for_this_sample.tokens_per_second
			num_gen_tokens = latest_response_for_this_sample.num_generated_tokens
			accumulated_text = (
				latest_response_for_this_sample.accumulated_text[-1]
				if latest_response_for_this_sample.accumulated_text
				else ""
			)
			time_spent = latest_response_for_this_sample.time_spent_computing

			for resp_step in sample_responses:
				text_segments_for_detok.extend(resp_step.text)
				all_token_ids.extend(resp_step.token_ids)

			# Detokenize the collected text segments for this step
			final_text_this_step = text_tokens_to_string(text_segments_for_detok)

			samples.append(
				ReturnSample(
					text=final_text_this_step,  # Detokenized text for this step
					token_ids=all_token_ids,  # All token IDs up to this point for this sample
					accumulated_text=accumulated_text,  # The full accumulated text so far
					time_spent_computing=time_spent,
					tokens_per_second=tps,
					num_generated_tokens=num_gen_tokens,
				)
			)
		return samples

	async def complete(
		self, request: vSurgeRequest
	) -> tp.AsyncGenerator[tp.List[ReturnSample], None]:
		"""
		Initiates and streams the results of a text completion request.

		This method creates an `ActiveRequest` from the `vSurgeRequest`,
		places it on the driver's prefill queue, and then asynchronously
		iterates through the results provided by the `ActiveRequest`'s
		`return_channel`.

		It handles both client-side and server-side tokenization scenarios.
		For server-side tokenization, it buffers responses if they end with
		incomplete multi-byte characters and processes them together when
		a complete sequence is available or at the end of generation.

		Args:
		    request (vSurgeRequest): The vSurgeRequest containing the prompt,
		        generation parameters, and tokenization preference.

		Yields:
		    tp.List[ReturnSample]: A list containing one `ReturnSample` per
		        concurrent generation (usually one, unless batching within
		        `complete` is supported and used). Each `ReturnSample` represents
		        a segment of the generated text or tokens for a single step.
		        - If `request.is_client_side_tokenization` is True, `ReturnSample.text`
		          will contain raw token strings (possibly byte tokens), and
		          `ReturnSample.token_ids` will be populated.
		        - If `request.is_client_side_tokenization` is False, `ReturnSample.text`
		          will contain detokenized text for the current step, and
		          `ReturnSample.accumulated_text` will contain the full detokenized
		          text generated so far.

		Raises:
		    RuntimeError: If the prefill queue is full when trying to place the request.
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
			stop=request.stop,
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

		buffered_response_list: list[list[ReturnSample]] = []
		async for response_step in active_request.return_channel:
			response_step = tp.cast(list[ReturnSample], response_step)
			if request.is_client_side_tokenization:
				yield self.process_client_side_tokenization_response(response_step)
			else:
				if self.should_buffer_response(response_step):
					buffered_response_list.append(response_step)
				else:
					if buffered_response_list:
						yield self.process_server_side_tokenization_response(
							response_step,
							buffered_response_list,
						)
						buffered_response_list = []
					else:
						yield self.process_server_side_tokenization_response(
							response_step,
							[],
						)

		if not request.is_client_side_tokenization and buffered_response_list:
			last_real_response_step = buffered_response_list[-1]
			dummy_final_response_step = [
				ReturnSample(
					text=[],
					token_ids=[],
					accumulated_text=s.accumulated_text,
					time_spent_computing=s.time_spent_computing,
					tokens_per_second=s.tokens_per_second,
					num_generated_tokens=s.num_generated_tokens,
				)
				for s in last_real_response_step
			]
			yield self.process_server_side_tokenization_response(
				dummy_final_response_step,
				buffered_response_list,
			)

	async def generate(
		self,
		prompts: tp.Union[str, tp.Sequence[str]],
		sampling_params: tp.Optional[
			tp.Union[SamplingParams, tp.Sequence[SamplingParams]]
		] = None,
		stream: bool = False,
	) -> tp.Union[tp.List[ReturnSample], tp.AsyncGenerator[tp.List[ReturnSample], None]]:
		"""
		Generates text completions concurrently for the given prompts.

		This method handles single or multiple prompts, applying corresponding
		sampling parameters. It can operate in streaming or batch mode.

		Args:
		    prompts (tp.Union[str, tp.Sequence[str]]): A single prompt string or a
		        sequence of prompt strings.
		    sampling_params (tp.Optional[tp.Union[SamplingParams, tp.Sequence[SamplingParams]]]):
		        - If None, default `SamplingParams` will be used for all prompts.
		        - If a single `SamplingParams` object, it will be applied to all prompts.
		        - If a sequence of `SamplingParams`, its length must match the
		          length of `prompts`.
		    stream (bool):
		        - If True: Returns an async generator. It yields `tp.List[ReturnSample]`
		          as soon as a generation step (e.g., one token or a chunk of text)
		          is completed for *any* of the concurrent requests. Each yielded list
		          typically contains one `ReturnSample` corresponding to the request that
		          produced output at that moment.
		        - If False: Returns a `tp.List[ReturnSample]`. It waits for all
		          requests to complete and then returns a list where each `ReturnSample`
		          contains the fully aggregated generated text for one corresponding
		          input prompt.

		Returns:
		    tp.Union[tp.List[ReturnSample], tp.AsyncGenerator[tp.List[ReturnSample], None]]:
		        - If `stream` is True: An async generator yielding lists of `ReturnSample`.
		        - If `stream` is False: A list of `ReturnSample` objects, one for each
		          input prompt, containing the complete generated text and final metrics.

		Raises:
		    ValueError: If `prompts` is not a string or sequence, or if the lengths
		        of `prompts` and `sampling_params` (when `sampling_params` is a
		        sequence) do not match.
		    RuntimeError: If the underlying driver's queue is full (propagated from `complete`).
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
	) -> tp.AsyncGenerator[tp.List[ReturnSample], None]:
		"""
		Helper for concurrent streaming generation.

		Creates an asyncio.Queue and runs `self.complete` for each request
		concurrently. Results from `self.complete` (which are `List[ReturnSample]`)
		are put into the queue and yielded as they arrive.

		Args:
		    requests (tp.List[vSurgeRequest]): A list of `vSurgeRequest` objects
		        to process concurrently.

		Yields:
		    tp.List[ReturnSample]: Each item yielded is the result of one generation
		        step from one of the concurrent `self.complete` calls. This list
		        typically contains a single `ReturnSample`.

		Raises:
		    Exception: Propagates exceptions from `self.complete` calls. If any
		        completion task fails, all other tasks are cancelled, and the
		        exception is raised.
		"""
		q: asyncio.Queue[tp.Union[tp.List[ReturnSample], Exception, object]] = (
			asyncio.Queue()
		)
		tasks = set()
		_SENTINEL = object()

		async def _run_completion(request: vSurgeRequest):
			"""Wraps self.complete to put results/exceptions into the shared queue."""
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

		finished_tasks_count = 0
		while finished_tasks_count < len(requests):
			item = await q.get()
			try:
				if item is _SENTINEL:
					finished_tasks_count += 1
				elif isinstance(item, Exception):
					for task_to_cancel in tasks:
						if not task_to_cancel.done():
							task_to_cancel.cancel()
					await asyncio.gather(*tasks, return_exceptions=True)
					raise item
				else:
					yield tp.cast(tp.List[ReturnSample], item)
			finally:
				q.task_done()

		await asyncio.gather(*tasks, return_exceptions=True)

	async def _generate_batch(
		self,
		requests: tp.List[vSurgeRequest],
	) -> tp.List[ReturnSample]:
		"""
		Helper for concurrent batch generation.

		Runs `self.complete` for each request concurrently and collects all
		generated steps. Then, for each request, it aggregates all its steps
		into a single `ReturnSample` representing the full generation.

		Args:
		    requests (tp.List[vSurgeRequest]): A list of `vSurgeRequest` objects
		        to process concurrently.

		Returns:
		    tp.List[ReturnSample]: A list where each `ReturnSample` corresponds to an
		        input request and contains the fully aggregated generated text,
		        all token IDs, average TPS, and total generated tokens.

		Raises:
		    Exception: Propagates exceptions from `self.complete` calls. If any
		        completion task fails, all other tasks are cancelled, and the
		        exception is raised.
		"""

		async def _collect_all_steps(request: vSurgeRequest) -> list[list[ReturnSample]]:
			"""Consumes the self.complete generator and returns all its yielded steps."""
			all_steps_for_one_request = []
			async for step_result_list in self.complete(request):
				# step_result_list is List[ReturnSample], usually one element for a single request
				all_steps_for_one_request.append(step_result_list)
			return all_steps_for_one_request

		tasks = [asyncio.create_task(_collect_all_steps(req)) for req in requests]
		try:
			# results_per_request will be:
			# [
			#   [[req1_step1_sample], [req1_step2_sample], ...], # All steps for request 1
			#   [[req2_step1_sample], [req2_step2_sample], ...], # All steps for request 2
			#   ...
			# ]
			# where each _sample is a ReturnSample, and each step_X_sample is wrapped in a list.
			results_per_request: list[list[list[ReturnSample]]] = await asyncio.gather(*tasks)
		except Exception as e:  # If any task failed
			for task in tasks:
				if not task.done():
					task.cancel()
			await asyncio.gather(*tasks, return_exceptions=True)  # Wait for cancellations
			raise e

		final_results: tp.List[ReturnSample] = []
		for request_output_steps in results_per_request:
			if not request_output_steps:
				final_results.append(
					ReturnSample(
						text="",
						token_ids=[],
						accumulated_text="",
						tokens_per_second=0.0,
						num_generated_tokens=0,
					)
				)
				continue

			last_step_list = request_output_steps[-1]
			if not last_step_list:
				final_results.append(
					ReturnSample(
						text="",
						token_ids=[],
						accumulated_text="",
						tokens_per_second=0.0,
						num_generated_tokens=0,
					)
				)
				continue

			final_sample_for_request = last_step_list[0]

			aggregated_text_parts = []
			aggregated_token_ids_parts = []

			for step_list in request_output_steps:
				if step_list:
					sample_in_step = step_list[0]
					aggregated_text_parts.append(sample_in_step.text)
					aggregated_token_ids_parts.extend(sample_in_step.token_ids)
			final_results.append(
				ReturnSample(
					text=final_sample_for_request.accumulated_text,
					token_ids=final_sample_for_request.token_ids,  # These are already accumulated by `complete`
					accumulated_text=final_sample_for_request.accumulated_text,  # Redundant for batch but consistent
					time_spent_computing=final_sample_for_request.time_spent_computing,  # from last sample
					tokens_per_second=final_sample_for_request.tokens_per_second,  # from last sample
					num_generated_tokens=final_sample_for_request.num_generated_tokens,  # from last sample
				)
			)
		return final_results
