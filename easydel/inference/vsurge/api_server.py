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
"""Implements a FastAPI server for serving vEngine models, mimicking OpenAI API."""

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
from .vsurge import vSurge, vSurgeRequest

TIMEOUT_KEEP_ALIVE = 5.0

APP = FastAPI(title="EasyDeL vSurge API Server")
logger = get_logger("vSurgeApiServer")


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


class vSurgeApiServer:
	"""
	FastAPI server for serving vEngine instances.

	This server provides endpoints mimicking the OpenAI API structure for chat completions,
	liveness/readiness checks, token counting, and listing available models.
	It handles both streaming and non-streaming requests asynchronously using a thread pool.
	"""

	def __init__(
		self,
		vsurge_map: tp.Union[tp.Dict[str, vSurge], vSurge] = None,
		max_workers: int = 10,
		oai_like_processor: bool = True,
	) -> None:
		if isinstance(vsurge_map, vSurge):
			vsurge_map = {vsurge_map.vsurge_name: vsurge_map}

		self.vsurge_map: tp.Dict[str, vSurge] = {}
		for name, vsurge in vsurge_map.items():
			err_msg = (
				f"Value for key '{name}' in vsurge_map must be an instance of `vSurge`, "
				f"got {type(vsurge).__name__} instead."
			)
			assert isinstance(vsurge, vSurge), err_msg
			self.vsurge_map[name] = vsurge
			logger.info(f"Added vsurge: {name}")

		self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
		self.logger = logger
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
		self,
		request: ChatCompletionRequest,
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
			vsurge = self._get_vsurge(request.model)

			# Process the prompt
			prompt = request.prompt
			if isinstance(prompt, list):
				prompt = prompt[0]

			if not request.stream:
				return await self._handle_completion_response_async(
					request,
					vsurge,
					prompt,
				)
			else:
				return await self._handle_completion_streaming_async(
					request,
					vsurge,
					prompt,
				)

		except Exception as e:
			self.logger.exception(f"Error during completion for model {request.model}: {e}")
			return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

	async def chat_completions(self, request: ChatCompletionRequest):
		"""
		Handles chat completion requests (POST /v1/chat/completions).

		Validates the request, retrieves the appropriate vEngine model,
		tokenizes the input, and delegates to streaming or non-streaming handlers.

		Args:
		    request (ChatCompletionRequest): The incoming request data.

		Returns:
		    Union[JSONResponse, StreamingResponse]: The generated response, either
		        a complete JSON object or a streaming event-stream.
		"""
		try:
			vsurge = self._get_vsurge(request.model)
			ids = await self._prepare_vsurge_input_async(request, vsurge)
			if not request.stream:
				return await self._handle_completion_response_async(request, vsurge, ids)
			else:
				return await self._handle_completion_streaming_async(request, vsurge, ids)
		except Exception as e:
			self.logger.exception(
				f"Error during chat completion for model {request.model}: {e}"
			)
			return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

	def _get_vsurge(self, model_name: str) -> vSurge:
		"""
		Retrieves the vEngine instance for the given model name.

		Args:
		    model_name (str): The requested model name.

		Returns:
		    vEngine: The corresponding vEngine instance.

		Raises:
		    RuntimeError: If the model name is not found in the `vsurge_map`.
		"""
		vsurge = self.vsurge_map.get(model_name)
		if vsurge is None:
			available_models = list(self.vsurge_map.keys())
			error_msg = (
				f"Invalid model name: '{model_name}'. Available models: {available_models}"
			)
			self.logger.error(error_msg)
			raise RuntimeError(error_msg)
		return vsurge

	def _prepare_vsurge_input(
		self,
		request: ChatCompletionRequest,
		vsurge: vSurge,
	) -> str:
		conversation = request.model_dump(exclude_unset=True)["messages"]
		processor = vsurge.processor
		if isinstance(processor, ProcessorMixin) and self.oai_like_processor:
			from easydel.trainers.prompt_utils import convert_to_openai_format

			conversation = convert_to_openai_format(conversation)
		try:
			return processor.apply_chat_template(
				conversation=conversation,
				add_generation_prompt=True,
				tokenize=False,
			)
		except Exception as e:
			self.logger.exception(
				f"Error applying chat template for model {vsurge.vsurge_name}: {e}"
			)
			raise RuntimeError(f"Error tokenizing input: {e}") from e

	async def _prepare_vsurge_input_async(self, request, vsurge) -> dict:
		"""Runs tokenization in the thread pool."""
		return await asyncio.get_event_loop().run_in_executor(
			self.thread_pool,
			self._prepare_vsurge_input,
			request,
			vsurge,
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

	async def _handle_completion_response_async(
		self,
		request: ChatCompletionRequest | CompletionRequest,
		vsurge: vSurge,
		content: str,
	):
		"""Runs the non-streaming handler in the thread pool."""
		"""
		Generates a complete, non-streaming chat response.

		Runs the vEngine generation loop to completion and formats the result.

		Args:
		    request (ChatCompletionRequest): The original request.
		    vsurge (vEngine): The vEngine instance.
		    ids (dict): The tokenized input dictionary.

		Returns:
		    ChatCompletionResponse: The complete chat response object.
		"""
		prompt_tokens = vsurge.count_tokens(content)
		sampling_params = self._create_sampling_params_from_request(request)

		response_state = None
		final_response = ""
		async for response_state in vsurge.complete(
			request=vSurgeRequest.from_sampling_params(
				prompt=content,
				sampling_params=sampling_params,
			)
		):
			final_response += response_state[0].text
		response_state = response_state[0]
		if response_state is None:
			raise RuntimeError("Generation failed to produce any output state.")

		generated_tokens = response_state.num_generated_tokens
		time_spent_computing = 0
		tokens_per_second = response_state.tokens_per_second

		finish_reason = (
			"length" if generated_tokens >= sampling_params.max_tokens else "stop"
		)

		function_call_result = None

		usage = self._create_usage_info(
			prompt_tokens=prompt_tokens,
			completion_tokens=generated_tokens,
			total_tokens=prompt_tokens + generated_tokens,
			time_spent_computing=time_spent_computing,
			tokens_per_second=tokens_per_second,
		)
		if isinstance(request, ChatCompletionRequest):
			return ChatCompletionResponse(
				model=request.model,
				choices=[
					ChatCompletionResponseChoice(
						index=generated_tokens + 1,
						message=ChatMessage(
							role="assistant",
							content=final_response,
							function_call=function_call_result,
						),
						finish_reason="function_call" if function_call_result else finish_reason,
					)
				],
				usage=usage,
			)
		elif isinstance(request, CompletionRequest):
			return CompletionResponse(
				model=request.model,
				choices=[
					CompletionResponseChoice(
						index=generated_tokens + 1,
						text=final_response,
						finish_reason=finish_reason,
					)
				],
				usage=usage,
			)
		else:
			raise NotImplementedError("UnKnown request type!")

	async def _handle_completion_streaming_async(
		self,
		request: ChatCompletionRequest | CompletionRequest,
		vsurge: vSurge,
		content: str,
	) -> StreamingResponse:
		"""Handle streaming response generation asynchronously."""

		async def stream_results() -> tp.AsyncGenerator[bytes, tp.Any]:
			prompt_tokens = vsurge.count_tokens(content)

			sampling_params = self._create_sampling_params_from_request(request)

			try:
				async for response_state in vsurge.complete(
					request=vSurgeRequest.from_sampling_params(
						prompt=content,
						sampling_params=sampling_params,
					)
				):
					response_state = response_state[0]
					chunk_usage = await self._create_usage_info_async(
						prompt_tokens=prompt_tokens,
						completion_tokens=response_state.num_generated_tokens,
						total_tokens=prompt_tokens + response_state.num_generated_tokens,
						time_spent_computing=0,
						tokens_per_second=response_state.tokens_per_second,
					)
					if isinstance(request, ChatCompletionRequest):
						stream_resp = ChatCompletionStreamResponse(
							model=request.model,
							choices=[
								ChatCompletionStreamResponseChoice(
									index=response_state.num_generated_tokens,
									delta=DeltaMessage(
										role="assistant"
										if response_state.num_generated_tokens == 0
										else None,
										content=response_state.text,
										function_call=None,
									),
									finish_reason=None,
								)
							],
							usage=chunk_usage,
						)
					elif isinstance(request, CompletionRequest):
						stream_resp = CompletionStreamResponse(
							model=request.model,
							choices=[
								CompletionStreamResponseChoice(
									text=response_state.text,
									index=response_state.num_generated_tokens,
									finish_reason=None,
								)
							],
							usage=chunk_usage,
						)
					else:
						raise NotImplementedError("UnKnown request type!")
					last_response_state = response_state

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
					if last_response_state.num_generated_tokens >= sampling_params.max_tokens
					else "stop"
				)

				final_usage = await self._create_usage_info_async(
					prompt_tokens=prompt_tokens,
					completion_tokens=last_response_state.num_generated_tokens,
					total_tokens=prompt_tokens + last_response_state.num_generated_tokens,
					time_spent_computing=0,
					tokens_per_second=last_response_state.tokens_per_second,
				)
				if isinstance(request, ChatCompletionRequest):
					final_resp = ChatCompletionStreamResponse(
						model=request.model,
						choices=[
							ChatCompletionStreamResponseChoice(
								index=last_response_state.num_generated_tokens + 1,
								delta=DeltaMessage(),
								finish_reason=finish_reason,
							)
						],
						usage=final_usage,
					)
				elif isinstance(request, CompletionRequest):
					final_resp = CompletionStreamResponse(
						model=request.model,
						choices=[
							CompletionStreamResponseChoice(
								text="",
								index=response_state.num_generated_tokens + 1,
								finish_reason=finish_reason,
							)
						],
						usage=final_usage,
					)
				else:
					raise NotImplementedError("UnKnown request type!")
				yield (
					"data: " + final_resp.model_dump_json(exclude_unset=True) + "\n\n"
				).encode("utf-8")
			else:
				self.logger.warning("Streaming finished without producing any response state.")

		return StreamingResponse(stream_results(), media_type="text/event-stream")

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
			for model_id in self.vsurge_map.keys()
		]
		return JSONResponse({"object": "list", "data": models_data}, status_code=200)

	async def count_tokens(self, request: CountTokenRequest):
		"""Token counting endpoint (POST /v1/count_tokens)."""
		try:
			conv = request.conversation
			model_name = request.model
			vsurge = self._get_vsurge(model_name)
			# Run token counting in thread pool as it might involve processing
			num_tokens = await asyncio.get_event_loop().run_in_executor(
				self.thread_pool,
				vsurge.count_tokens,
				self._prepare_vsurge_input(conv, vsurge),
			)
			return JSONResponse({"model": model_name, "count": num_tokens}, status_code=200)
		except Exception as e:
			self.logger.exception(f"Error counting tokens for model {request.model}: {e}")
			return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

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
