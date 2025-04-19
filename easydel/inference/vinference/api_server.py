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
"""Implements a FastAPI server for serving vInference models, mimicking OpenAI API."""

from __future__ import annotations

import asyncio
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus

import uvicorn
from eformer.pytree import auto_pytree
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import ProcessorMixin

from easydel.inference.utilities import SamplingParams
from easydel.utils.helpers import get_logger
from easydel.utils.lazy_import import is_package_available

from ..openai_api_modules import (
	ChatCompletionRequest,
	ChatCompletionResponse,
	ChatCompletionResponseChoice,
	ChatCompletionStreamResponse,
	ChatCompletionStreamResponseChoice,
	ChatMessage,
	CompletionRequest,
	CompletionResponse,
	CompletionResponseChoice,
	CompletionStreamResponse,
	CompletionStreamResponseChoice,
	CountTokenRequest,
	DeltaMessage,
	UsageInfo,
)

if tp.TYPE_CHECKING:
	from ..vinference import vInference, vInferenceConfig
else:
	vInference = tp.Any
	vInferenceConfig = tp.Any
TIMEOUT_KEEP_ALIVE = 5.0

APP = FastAPI(title="EasyDeL vInference API Server")
logger = get_logger("vInferenceApiServer")


@auto_pytree
class EndpointConfig:
	"""Configuration for a FastAPI endpoint."""

	path: str
	handler: tp.Callable
	methods: list[str]
	summary: tp.Optional[str] = None
	tags: tp.Optional[list[str]] = None


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
	"""Creates a standardized JSON error response."""
	return JSONResponse({"error": {"message": message}}, status_code=status_code.value)


class vInferenceApiServer:
	"""
	FastAPI server for serving vInference instances.

	This server provides endpoints mimicking the OpenAI API structure for chat completions,
	liveness/readiness checks, token counting, and listing available models.
	It handles both streaming and non-streaming requests asynchronously using a thread pool.
	"""

	def __init__(
		self,
		inference_map: tp.Union[tp.Dict[str, vInference], vInference] = None,
		inference_init_call: tp.Optional[tp.Callable[[], vInference]] = None,
		max_workers: int = 10,
		allow_parallel_workload: bool = False,
		oai_like_processor: bool = True,
	) -> None:
		"""
		Initializes the vInferenceApiServer.

		Args:
		    inference_map (tp.Union[tp.Dict[str, vInference], vInference], optional):
		        Either a dictionary mapping model names (str) to initialized `vInference`
		        instances or a single `vInference` instance. If a single instance is
		        provided, its `inference_name` will be used as the key.
		        Defaults to None.
		    inference_init_call (tp.Optional[tp.Callable[[], vInference]], optional):
		        A callable that returns an initialized `vInference` instance or a dictionary
		        as described above. This is useful for lazy initialization.
		        Defaults to None. Either `inference_map` or `inference_init_call` must be provided.
		    max_workers (int): The maximum number of worker threads in the thread pool
		        for handling inference requests. Defaults to 10.
				oai_like_processor (bool): automatically upcast processor conversation to openai format.
		Raises:
		    AssertionError: If neither `inference_map` nor `inference_init_call` is provided,
		        or if the provided values are not of the expected `vInference` type.
		"""
		from ..vinference import vInference

		if inference_init_call is not None:
			inference_map = inference_init_call()

		assert inference_map is not None, (
			"`inference_map` or `inference_init_call` must be provided."
		)
		if isinstance(inference_map, vInference):
			# If a single vInference instance is given, wrap it in a dict
			inference_map = {inference_map.inference_name: inference_map}

		self.inference_map: tp.Dict[str, vInference] = {}
		for name, inference in inference_map.items():
			err_msg = (
				f"Value for key '{name}' in inference_map must be an instance of `vInference`, "
				f"got {type(inference).__name__} instead."
			)
			assert isinstance(inference, vInference), err_msg
			self.inference_map[name] = inference
			logger.info(f"Added inference model: {name}")

		self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
		self.logger = logger
		self.allow_parallel_workload = allow_parallel_workload
		self.oai_like_processor = oai_like_processor
		self._register_endpoints()

	@property
	def _endpoints(self) -> tp.List[EndpointConfig]:
		"""Defines all API endpoints for the server."""
		return [
			EndpointConfig(
				path="/v1/chat/completions",
				handler=self.chat_completions,
				methods=["POST"],
				tags=["Chat"],
				summary="Creates a model response for the given chat conversation.",
			),
			EndpointConfig(
				path="/v1/completions",
				handler=self.completions,
				methods=["POST"],
				tags=["Completions"],
				summary="Creates a completion for the provided prompt.",
			),
			EndpointConfig(
				path="/liveness",
				handler=self.liveness,
				methods=["GET"],
				tags=["Health"],
				summary="Checks if the API server is running.",
			),
			EndpointConfig(
				path="/readiness",
				handler=self.readiness,
				methods=["GET"],
				tags=["Health"],
				summary="Checks if the API server is ready to receive requests.",
			),
			EndpointConfig(
				path="/v1/count_tokens",  # Changed path for consistency
				handler=self.count_tokens,
				methods=["POST"],
				tags=["Utility"],
				summary="Counts the number of tokens in a given text or conversation.",
			),
			EndpointConfig(
				path="/v1/models",  # Changed path to match OpenAI standard
				handler=self.available_inference,
				methods=["GET"],
				tags=["Utility"],
				summary="Lists the models available through this API.",
			),
		]

	def _create_sampling_params_from_request(
		self, request: ChatCompletionRequest
	) -> SamplingParams:
		"""Creates SamplingParams from a ChatCompletionRequest."""
		return SamplingParams(
			max_tokens=int(request.max_tokens),
			temperature=float(request.temperature),
			presence_penalty=float(request.presence_penalty),
			frequency_penalty=float(request.frequency_penalty),
			repetition_penalty=float(request.repetition_penalty),
			top_k=int(request.top_k),
			top_p=float(request.top_p),
			min_p=float(request.min_p),
			suppress_tokens=request.suppress_tokens,
		)

	async def chat_completions(self, request: ChatCompletionRequest):
		"""
		Handles chat completion requests (POST /v1/chat/completions).

		Validates the request, retrieves the appropriate vInference model,
		tokenizes the input, and delegates to streaming or non-streaming handlers.

		Args:
		    request (ChatCompletionRequest): The incoming request data.

		Returns:
		    Union[JSONResponse, StreamingResponse]: The generated response, either
		        a complete JSON object or a streaming event-stream.
		"""
		try:
			inference = self._get_inference_model(request.model)
			ids = await self._prepare_tokenized_input_async(request, inference)
			if not request.stream:
				return await self._handle_non_streaming_response_async(request, inference, ids)
			else:
				return await self._handle_streaming_response_async(request, inference, ids)
		except Exception as e:
			self.logger.exception(
				f"Error during chat completion for model {request.model}: {e}"
			)
			return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

	def _get_inference_model(self, model_name: str) -> vInference:
		"""
		Retrieves the vInference instance for the given model name.

		Args:
		    model_name (str): The requested model name.

		Returns:
		    vInference: The corresponding vInference instance.

		Raises:
		    RuntimeError: If the model name is not found in the `inference_map`.
		"""
		inference = self.inference_map.get(model_name)
		if inference is None:
			available_models = list(self.inference_map.keys())
			error_msg = (
				f"Invalid model name: '{model_name}'. Available models: {available_models}"
			)
			self.logger.error(error_msg)
			raise RuntimeError(error_msg)
		return inference

	def _prepare_tokenized_input(
		self,
		request: ChatCompletionRequest,
		inference: vInference,
	) -> dict:
		"""
		Tokenizes the input conversation using the appropriate processor.

		Args:
		    request (ChatCompletionRequest): The request containing the messages.
		    inference (vInference): The vInference instance with the tokenizer.

		Returns:
		    dict: A dictionary containing the tokenized input IDs and potentially
		        other model inputs (like attention mask).
		"""
		conversation = request.model_dump(exclude_unset=True)["messages"]
		processor = inference.processor_class
		if isinstance(processor, ProcessorMixin) and self.oai_like_processor:
			from easydel.trainers.prompt_utils import convert_to_openai_format

			conversation = convert_to_openai_format(conversation)
		try:
			return processor.apply_chat_template(
				conversation=conversation,
				return_tensors="np",
				add_generation_prompt=True,
				return_dict=True,
				tokenize=True,
				padding_side="left",
			)
		except Exception as e:
			self.logger.exception(
				f"Error applying chat template for model {inference.inference_name}: {e}"
			)
			raise RuntimeError(f"Error tokenizing input: {e}") from e

	async def _prepare_tokenized_input_async(self, request, inference) -> dict:
		"""Runs tokenization in the thread pool."""
		return await asyncio.get_event_loop().run_in_executor(
			self.thread_pool,
			self._prepare_tokenized_input,
			request,
			inference,
		)

	def _create_usage_info(
		self,
		prompt_tokens: int,
		completion_tokens: int,
		total_tokens: int,
		time_spent_computing: float,
		tokens_per_second: float,
	) -> UsageInfo:
		"""Creates a UsageInfo object."""
		return UsageInfo(
			prompt_tokens=prompt_tokens,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
			tokens_per_second=tokens_per_second,
			processing_time=time_spent_computing,
		)

	def _handle_non_streaming_response(
		self,
		request: ChatCompletionRequest,
		inference: vInference,
		ids: dict,
	) -> ChatCompletionResponse:
		"""
		Generates a complete, non-streaming chat response.

		Runs the vInference generation loop to completion and formats the result.

		Args:
		    request (ChatCompletionRequest): The original request.
		    inference (vInference): The vInference instance.
		    ids (dict): The tokenized input dictionary.

		Returns:
		    ChatCompletionResponse: The complete chat response object.
		"""
		prompt_tokens = ids["input_ids"].shape[-1]
		sampling_params = self._create_sampling_params_from_request(request)

		# Generate response - loop until the last response is captured
		response_state = None
		for response_state in inference.generate(**ids, sampling_params=sampling_params):
			pass  # Keep iterating until the generator is exhausted

		if response_state is None:
			raise RuntimeError("Generation failed to produce any output state.")

		final_sequences = response_state.sequences
		generated_tokens = response_state.generated_tokens
		time_spent_computing = response_state._time_spent_computing
		tokens_per_second = response_state.tokens_per_second
		padded_length = response_state.padded_length

		# Decode the generated part of the sequences
		final_responses = inference.tokenizer.batch_decode(
			final_sequences[..., padded_length:],
			skip_special_tokens=True,
		)

		# Determine finish reason
		finish_reason = (
			"length"
			if generated_tokens >= inference.generation_config.max_new_tokens
			else "stop"
		)

		# Check if function calling is requested
		function_call_result = None
		if hasattr(request, "functions") and request.functions:
			# Process function call from response if present
			from json import JSONDecodeError, loads

			for response in final_responses:
				try:
					# Look for function call patterns in the response
					if "{" in response and "}" in response:
						possible_json = response[response.find("{") : response.rfind("}") + 1]
						parsed = loads(possible_json)
						if "name" in parsed and ("arguments" in parsed or "params" in parsed):
							function_call_result = {
								"name": parsed.get("name"),
								"arguments": parsed.get("arguments", parsed.get("params", "{}")),
							}
							break
				except JSONDecodeError:
					continue

		usage = self._create_usage_info(
			prompt_tokens=prompt_tokens,
			completion_tokens=generated_tokens,
			total_tokens=prompt_tokens + generated_tokens,
			time_spent_computing=time_spent_computing,
			tokens_per_second=tokens_per_second,
		)

		return ChatCompletionResponse(
			model=request.model,
			choices=[
				ChatCompletionResponseChoice(
					index=i,
					message=ChatMessage(
						role="assistant",
						content=final_response if not function_call_result else None,
						function_call=function_call_result,
					),
					finish_reason="function_call" if function_call_result else finish_reason,
				)
				for i, final_response in enumerate(final_responses)
			],
			usage=usage,
		)

	async def _handle_non_streaming_response_async(self, request, inference, ids):
		"""Runs the non-streaming handler in the thread pool."""
		return await asyncio.get_event_loop().run_in_executor(
			self.thread_pool,
			self._handle_non_streaming_response,
			request,
			inference,
			ids,
		)

	async def _handle_streaming_response_async(
		self,
		request: ChatCompletionRequest,
		inference: vInference,
		ids: tp.Dict,
	) -> StreamingResponse:
		"""Handle streaming response generation asynchronously."""

		async def stream_results() -> tp.AsyncGenerator[bytes, tp.Any]:
			prompt_tokens = inference.count_tokens(
				request.model_dump()["messages"],
				oai_like=self.oai_like_processor,
			)

			prompt_tokens = ids["input_ids"].shape[-1]
			sampling_params = self._create_sampling_params_from_request(request)

			def _blocking_generator():
				"""The actual blocking generation call."""
				yield from inference.generate(**ids, sampling_params=sampling_params)

			async def _generate():
				return await asyncio.get_event_loop().run_in_executor(
					self.thread_pool,
					_blocking_generator,
				)

			padded_sequence_length = None
			current_token_index = 0
			last_response_state = None

			try:
				async for response_state in self._aiter_generator(await _generate()):
					if padded_sequence_length is None:
						padded_sequence_length = response_state.padded_length
					next_slice = slice(
						padded_sequence_length,
						padded_sequence_length + inference.generation_config.streaming_chunks,
					)
					padded_sequence_length += inference.generation_config.streaming_chunks

					decoded_responses = await asyncio.get_event_loop().run_in_executor(
						self.thread_pool,
						inference.tokenizer.batch_decode,
						response_state.sequences[..., next_slice],
						True,  # skip_special_tokens
					)

					chunk_usage = await self._create_usage_info_async(
						prompt_tokens=prompt_tokens,
						completion_tokens=response_state.generated_tokens,
						total_tokens=prompt_tokens + response_state.generated_tokens,
						time_spent_computing=response_state._time_spent_computing,
						tokens_per_second=response_state.tokens_per_second,
					)

					stream_resp = ChatCompletionStreamResponse(
						model=request.model,
						choices=[
							ChatCompletionStreamResponseChoice(
								index=current_token_index,
								delta=DeltaMessage(
									role="assistant" if current_token_index == 0 else None,
									content=decoded_response,
									function_call=None,
								),
								finish_reason=None,
							)
							for _, decoded_response in enumerate(decoded_responses)
						],
						usage=chunk_usage,
					)
					last_response_state = response_state
					current_token_index += 1

					yield (
						"data: " + stream_resp.model_dump_json(exclude_unset=True) + "\n\n"
					).encode("utf-8")

			except Exception as e:
				self.logger.exception(f"Error during streaming generation: {e}")
				yield (
					f"data: {create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e)).body.decode()}"  # type: ignore
					+ "\n\n"
				).encode("utf-8")
				return

			if last_response_state is not None:
				finish_reason = (
					"length"
					if last_response_state.generated_tokens
					>= inference.generation_config.max_new_tokens
					else "stop"
				)

				final_usage = await self._create_usage_info_async(
					prompt_tokens=prompt_tokens,
					completion_tokens=last_response_state.generated_tokens,
					total_tokens=prompt_tokens + last_response_state.generated_tokens,
					time_spent_computing=last_response_state._time_spent_computing,
					tokens_per_second=last_response_state.tokens_per_second,
				)
				final_resp = ChatCompletionStreamResponse(
					model=request.model,
					choices=[
						ChatCompletionStreamResponseChoice(
							index=current_token_index,
							delta=DeltaMessage(),  # Empty delta for final chunk
							finish_reason=finish_reason,
						)
						for _ in decoded_responses
					],
					usage=final_usage,
				)
				yield (
					"data: " + final_resp.model_dump_json(exclude_unset=True) + "\n\n"
				).encode("utf-8")
			else:
				self.logger.warning("Streaming finished without producing any response state.")

		return StreamingResponse(stream_results(), media_type="text/event-stream")

	async def _aiter_generator(self, generator):
		"""Convert a regular generator to an async generator."""
		for item in generator:
			yield item
			if self.allow_parallel_workload:
				# give other threads a chance to work
				await asyncio.sleep(0)

	async def _create_usage_info_async(
		self,
		prompt_tokens: int,
		completion_tokens: int,
		total_tokens: int,
		time_spent_computing: float,
		tokens_per_second: float,
	) -> UsageInfo:
		"""Async helper to create UsageInfo, potentially offloading if needed."""
		# Currently, UsageInfo creation is trivial, so no need to offload.
		# If it became complex, we could use run_in_executor here.
		return self._create_usage_info(
			prompt_tokens,
			completion_tokens,
			total_tokens,
			time_spent_computing,
			tokens_per_second,
		)

	async def liveness(self):
		"""Liveness check endpoint (GET /liveness)."""
		return JSONResponse({"status": "alive"}, status_code=200)

	async def readiness(self):
		"""Readiness check endpoint (GET /readiness)."""
		# Basic check: server is running. Could be extended to check model loading status.
		return JSONResponse({"status": "ready"}, status_code=200)

	async def available_inference(self):
		"""Lists available models (GET /v1/models)."""
		models_data = [
			{
				"id": model_id,
				"object": "model",
				"owned_by": "easydel",  # Or customize as needed
				"permission": [],
			}
			for model_id in self.inference_map.keys()
		]
		return JSONResponse({"object": "list", "data": models_data}, status_code=200)

	async def count_tokens(self, request: CountTokenRequest):
		"""Token counting endpoint (POST /v1/count_tokens)."""
		try:
			conv = request.conversation
			model_name = request.model
			inference = self._get_inference_model(model_name)

			# Run token counting in thread pool as it might involve processing
			num_tokens = await asyncio.get_event_loop().run_in_executor(
				self.thread_pool, inference.count_tokens, conv
			)
			return JSONResponse({"model": model_name, "count": num_tokens}, status_code=200)
		except Exception as e:
			self.logger.exception(f"Error counting tokens for model {request.model}: {e}")
			return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

	async def completions(self, request: CompletionRequest):
		"""
		Handles completion requests (POST /v1/completions).

		Processes the prompt for completion and returns generated text.

		Args:
			request (CompletionRequest): The incoming request data.

		Returns:
			Union[JSONResponse, StreamingResponse]: The generated response.
		"""
		try:
			inference = self._get_inference_model(request.model)

			# Process the prompt
			prompt = request.prompt
			if isinstance(prompt, list):
				prompt = prompt[0]  # Take the first prompt if multiple are provided

			# Tokenize the prompt
			inputs = inference.tokenizer(prompt, return_tensors="np")

			if not request.stream:
				return await self._handle_completion_response_async(request, inference, inputs)
			else:
				return await self._handle_completion_streaming_async(request, inference, inputs)

		except Exception as e:
			self.logger.exception(f"Error during completion for model {request.model}: {e}")
			return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

	def _handle_completion_response(
		self,
		request: CompletionRequest,
		inference: vInference,
		inputs: dict,
	) -> CompletionResponse:
		"""
		Generates a complete, non-streaming completion response.
		"""
		prompt_tokens = inputs["input_ids"].shape[-1]
		sampling_params = self._create_sampling_params_from_request(request)

		# Generate response
		response_state = None
		for response_state in inference.generate(**inputs, sampling_params=sampling_params):
			pass

		if response_state is None:
			raise RuntimeError("Generation failed to produce any output state.")

		final_sequences = response_state.sequences
		generated_tokens = response_state.generated_tokens
		time_spent_computing = response_state._time_spent_computing
		tokens_per_second = response_state.tokens_per_second
		padded_length = response_state.padded_length

		# Decode only the generated part
		completions = inference.tokenizer.batch_decode(
			final_sequences[..., padded_length:],
			skip_special_tokens=True,
		)

		finish_reason = (
			"length"
			if generated_tokens >= inference.generation_config.max_new_tokens
			else "stop"
		)

		usage = self._create_usage_info(
			prompt_tokens=prompt_tokens,
			completion_tokens=generated_tokens,
			total_tokens=prompt_tokens + generated_tokens,
			time_spent_computing=time_spent_computing,
			tokens_per_second=tokens_per_second,
		)

		return CompletionResponse(
			model=request.model,
			choices=[
				CompletionResponseChoice(
					text=completion,
					index=i,
					finish_reason=finish_reason,
				)
				for i, completion in enumerate(completions)
			],
			usage=usage,
		)

	async def _handle_completion_response_async(self, request, inference, inputs):
		"""Runs the non-streaming completion handler in the thread pool."""
		return await asyncio.get_event_loop().run_in_executor(
			self.thread_pool,
			self._handle_completion_response,
			request,
			inference,
			inputs,
		)

	async def _handle_completion_streaming_async(
		self,
		request: CompletionRequest,
		inference: vInference,
		inputs: dict,
	) -> StreamingResponse:
		"""
		Generates a streaming completion response.
		"""

		async def stream_results() -> tp.AsyncGenerator[bytes, None]:
			prompt_tokens = inputs["input_ids"].shape[-1]
			sampling_params = self._create_sampling_params_from_request(request)

			def _blocking_generator():
				yield from inference.generate(**inputs, sampling_params=sampling_params)

			async def _generate():
				return await asyncio.get_event_loop().run_in_executor(
					self.thread_pool,
					_blocking_generator,
				)

			current_token_index = 0
			last_response_state = None

			try:
				async for response_state in self._aiter_generator(await _generate()):
					last_response_state = response_state
					new_sequences = response_state.sequences
					padded_length = response_state.padded_length
					generated_tokens_so_far = response_state.generated_tokens
					stream_chunk_size = inference.generation_config.streaming_chunks

					# Get new tokens in this chunk
					start_slice = padded_length + current_token_index * stream_chunk_size
					end_slice = min(start_slice + stream_chunk_size, new_sequences.shape[-1])
					token_slice = new_sequences[..., start_slice:end_slice]

					if token_slice.shape[-1] == 0:
						continue

					decoded_responses = await asyncio.get_event_loop().run_in_executor(
						self.thread_pool,
						inference.tokenizer.batch_decode,
						token_slice,
						True,  # skip_special_tokens
					)

					chunk_usage = await self._create_usage_info_async(
						prompt_tokens=prompt_tokens,
						completion_tokens=generated_tokens_so_far,
						total_tokens=prompt_tokens + generated_tokens_so_far,
						time_spent_computing=response_state._time_spent_computing,
						tokens_per_second=response_state.tokens_per_second,
					)

					stream_resp = CompletionStreamResponse(
						model=request.model,
						choices=[
							CompletionStreamResponseChoice(
								text=decoded_response,
								index=i,
								finish_reason=None,
							)
							for i, decoded_response in enumerate(decoded_responses)
						],
						usage=chunk_usage,
					)
					current_token_index += 1
					yield (
						"data: " + stream_resp.model_dump_json(exclude_unset=True) + "\n\n"
					).encode("utf-8")

			except Exception as e:
				self.logger.exception(f"Error during streaming completion: {e}")
				yield (
					f"data: {create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e)).body.decode()}"
					+ "\n\n"
				).encode("utf-8")
				return

			# Final response chunk with finish reason
			if last_response_state is not None:
				finish_reason = (
					"length"
					if last_response_state.generated_tokens
					>= inference.generation_config.max_new_tokens
					else "stop"
				)
				final_usage = await self._create_usage_info_async(
					prompt_tokens=prompt_tokens,
					completion_tokens=last_response_state.generated_tokens,
					total_tokens=prompt_tokens + last_response_state.generated_tokens,
					time_spent_computing=last_response_state._time_spent_computing,
					tokens_per_second=last_response_state.tokens_per_second,
				)
				final_resp = CompletionStreamResponse(
					model=request.model,
					choices=[
						CompletionStreamResponseChoice(
							text="",  # Empty text for final chunk
							index=i,
							finish_reason=finish_reason,
						)
						for i in range(last_response_state.sequences.shape[0])
					],
					usage=final_usage,
				)
				yield (
					"data: " + final_resp.model_dump_json(exclude_unset=True) + "\n\n"
				).encode("utf-8")
			else:
				self.logger.warning("Streaming finished without producing any response state.")

			yield ("data: [DONE]\n\n").encode("utf-8")

		return StreamingResponse(stream_results(), media_type="text/event-stream")

	def _register_endpoints(self):
		"""Registers all defined API endpoints with the FastAPI application."""
		for endpoint in self._endpoints:
			# The handler needs to be wrapped to be recognized correctly by FastAPI's decorators
			# when defined within a class.
			APP.add_api_route(
				path=endpoint.path,
				endpoint=endpoint.handler,
				methods=endpoint.methods,
				summary=endpoint.summary,
				tags=endpoint.tags,
				response_model=None,  # Let FastAPI infer or handle manually
			)

	def fire(
		self,
		host="0.0.0.0",
		port=11556,
		metrics_port: tp.Optional[int] = None,
		log_level="info",  # Changed default log level
		ssl_keyfile: tp.Optional[str] = None,
		ssl_certfile: tp.Optional[str] = None,
	):
		"""
		Starts the uvicorn server to run the FastAPI application.

		Args:
		    host (str): The host address to bind to. Defaults to "0.0.0.0".
		    port (int): The port to listen on. Defaults to 11556.
		    metrics_port (tp.Optional[int]): The port for the Prometheus metrics server.
		        If None, defaults to `port + 1`. Set to -1 to disable.
		    log_level (str): The logging level for uvicorn. Defaults to "info".
		    ssl_keyfile (tp.Optional[str]): Path to the SSL key file for HTTPS.
		    ssl_certfile (tp.Optional[str]): Path to the SSL certificate file for HTTPS.
		"""
		if metrics_port is None:
			metrics_port = port + 1

		if metrics_port > 0 and is_package_available("prometheus_client"):
			try:
				from prometheus_client import start_http_server  # type:ignore

				start_http_server(metrics_port)
				self.logger.info(f"Prometheus metrics server started on port {metrics_port}")
			except Exception as e:
				self.logger.error(f"Failed to start Prometheus metrics server: {e}")
		elif metrics_port > 0:
			self.logger.warning(
				"Prometheus metrics requested but `prometheus_client` is not installed. "
				"Metrics server will not start. Install with `pip install prometheus-client`."
			)

		uvicorn_config = {
			"host": host,
			"port": port,
			"log_level": log_level,
			"timeout_keep_alive": TIMEOUT_KEEP_ALIVE,
		}

		# Use uvloop if available for better performance
		try:
			import uvloop  # type:ignore #noqa

			uvicorn_config["loop"] = "uvloop"
			self.logger.info("Using uvloop for the event loop.")
		except ImportError:
			self.logger.info("uvloop not found, using default asyncio event loop.")

		if ssl_keyfile and ssl_certfile:
			uvicorn_config["ssl_keyfile"] = ssl_keyfile
			uvicorn_config["ssl_certfile"] = ssl_certfile
			self.logger.info(f"Running with HTTPS enabled on {host}:{port}")
		else:
			self.logger.info(f"Running with HTTP on {host}:{port}")

		uvicorn.run(APP, **uvicorn_config)
