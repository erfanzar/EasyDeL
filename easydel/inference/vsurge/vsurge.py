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

import dataclasses
import datetime
import queue
import time
import typing as tp

import jax

from easydel.inference.utilities import SamplingParams
from easydel.utils.helpers import get_logger
from .driver import vDriver, AsyncMultifuture, ActiveRequest, ActiveRequestMetadata
from .engine import vEngine
from .utils import ReturnSample, is_byte_token, text_tokens_to_string

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
		del buffered_response_list
