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
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import AsyncGenerator, Callable, Dict, Optional

import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import start_http_server
from easydel.etils.etils import get_logger
from easydel.inference.vinference.api_models import (
	ChatCompletionRequest,
	ChatCompletionResponse,
	ChatCompletionResponseChoice,
	ChatCompletionStreamResponse,
	ChatCompletionStreamResponseChoice,
	CountTokenRequest,
	UsageInfo,
)

TIMEOUT_KEEP_ALIVE = 5.0


@dataclass
class EndpointConfig:
	path: str
	handler: Callable
	methods: list[str]
	summary: Optional[str] = None
	tags: Optional[list[str]] = None


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
	return JSONResponse({"message": message}, status_code=status_code.value)


class vInferenceApiServer:
	def __init__(
		self,
		inference_map: Dict[str, "vInference"] = None,  # noqa #type:ignore
	) -> None:
		from easydel.inference.vinference import vInference

		assert inference_map is not None, "`inference_map` can not be None."
		for inference in inference_map.values():
			assert isinstance(
				inference, vInference
			), "values and inferences in inference_map must be `vInference`"

		self.inference_map = inference_map
		self.router = APIRouter()
		self._endpoints = [
			EndpointConfig(
				path="/v1/chat/completions",
				handler=self.chat_completions,
				methods=["POST"],
				tags=["chat"],
				summary="Create a chat completion",
			),
			EndpointConfig(
				path="/liveness",
				handler=self.liveness,
				methods=["GET", "POST"],
				tags=["health"],
				summary="Check if ApiServer is up",
			),
			EndpointConfig(
				path="/readiness",
				handler=self.readiness,
				methods=["GET", "POST"],
				tags=["health"],
				summary="Check if ApiServer is ready",
			),
			EndpointConfig(
				path="/count_tokens",
				handler=self.count_tokens,
				methods=["POST"],
				tags=["utility"],
				summary="Count number of tokens with given inference runtime",
			),
			EndpointConfig(
				path="/available_inference",
				handler=self.available_inference,
				methods=["GET", "POST"],
				tags=["utility"],
				summary="Get available inference modules for requesting",
			),
		]
		self.app = FastAPI()
		self.server = uvicorn.Server(uvicorn.Config(app=self.app))
		self.logger = get_logger(__name__)
		self.patch_endpoints()

	async def chat_completions(self, request: ChatCompletionRequest):
		try:
			inference = self._get_inference_model(request.model)
			ids = self._prepare_tokenized_input(request.messages, inference)
			if not request.stream:
				return await self._handle_non_streaming_response(request, inference, ids)
			else:
				return await self._handle_streaming_response(request, inference, ids)

		except Exception as e:
			return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))

	def _get_inference_model(self, model_name: str) -> "vInference":  # noqa #type:ignore
		"""Get and validate inference model."""
		inference = self.inference_map.get(model_name)
		if inference is None:
			raise RuntimeError(f"Invalid model name: {model_name} is not available")
		return inference

	def _prepare_tokenized_input(self, messages, inference: "vInference") -> dict:  # noqa #type:ignore
		"""Prepare tokenized input for the model."""
		return inference.tokenizer.apply_chat_template(
			conversation=messages,
			return_dict=True,
			tokenize=True,
			return_tensors="np",
			add_generation_prompt=True,
			max_length=inference.model_prefill_length,
			padding="max_length",
		)

	def _create_usage_info(
		self,
		prefiled_length: int,
		ngenerated_tokens: int,
		processing_time: float,
		first_iter_flops: float,
		iter_flops: float,
	) -> UsageInfo:
		"""Create usage information."""
		return UsageInfo(
			first_iter_flops=first_iter_flops,
			iter_flops=iter_flops,
			prompt_tokens=prefiled_length,
			completion_tokens=ngenerated_tokens,
			total_tokens=ngenerated_tokens + prefiled_length,
			tps=ngenerated_tokens / processing_time,
			processing_time=processing_time,
		)

	async def _handle_non_streaming_response(
		self,
		request: ChatCompletionRequest,
		inference: "vInference",  # noqa #type:ignore
		ids: dict,
	) -> ChatCompletionResponse:
		"""Handle non-streaming response generation."""
		start = time.perf_counter()

		# Generate response
		async for response in inference.generate(
			input_ids=ids["input_ids"],
			attention_mask=ids["attention_mask"],
		):
			pass  # Keep last response

		processing_time = time.perf_counter() - start

		final_response = inference.tokenizer.decode(
			response.sequences[0][inference.model_prefill_length :],
			skip_special_tokens=True,
		)

		# Determine finish reason
		finish_reason = (
			"length"
			if response.generated_tokens == inference.generation_config.max_new_tokens
			else "stop"
		)

		return ChatCompletionResponse(
			model=request.model,
			choices=[
				ChatCompletionResponseChoice(
					response=final_response,
					finish_reason=finish_reason,
				)
			],
			usage=self._create_usage_info(
				inference.model_prefill_length,
				response.generated_tokens,
				processing_time,
				response.generate_func_flops,
				response.interval_func_flops,
			),
		)

	async def _handle_streaming_response(
		self,
		request: ChatCompletionRequest,
		inference: "vInference",  # noqa #type:ignore
		ids: dict,
	) -> StreamingResponse:
		"""Handle streaming response generation."""

		async def stream_results() -> AsyncGenerator[bytes, None]:
			start = time.perf_counter()
			padded_sequence_length = inference.model_prefill_length

			async for response in inference.generate(
				input_ids=ids["input_ids"],
				attention_mask=ids["attention_mask"],
			):
				next_slice = slice(
					padded_sequence_length,
					padded_sequence_length + inference.generation_config.streaming_chunks,
				)
				padded_sequence_length += inference.generation_config.streaming_chunks

				processing_time = time.perf_counter() - start

				stream_resp = ChatCompletionStreamResponse(
					model=request.model,
					choices=[
						ChatCompletionStreamResponseChoice(
							response=inference.tokenizer.decode(
								response.sequences[0][next_slice],
								skip_special_tokens=True,
							),
							finish_reason=None,
						)
					],
					usage=self._create_usage_info(
						inference.model_prefill_length,
						response.generated_tokens,
						processing_time,
						response.generate_func_flops,
						response.interval_func_flops,
					),
				)

				yield ("data: " + stream_resp.model_dump_json() + "\n\n").encode("utf-8")
			finish_reason = (
				"length"
				if response.generated_tokens == inference.generation_config.max_new_tokens
				else "stop"
			)
			stream_resp = ChatCompletionStreamResponse(
				model=request.model,
				choices=[
					ChatCompletionStreamResponseChoice(
						response="",
						finish_reason=finish_reason,
					)
				],
				usage=self._create_usage_info(
					inference.model_prefill_length,
					response.generated_tokens,
					processing_time,
					response.generate_func_flops,
					response.interval_func_flops,
				),
			)

			yield ("data: " + stream_resp.model_dump_json() + "\n\n").encode("utf-8")

		return StreamingResponse(stream_results(), media_type="text/event-stream")

	def liveness(self):
		return JSONResponse({"status": "ok"}, status_code=200)

	def readiness(self):
		return JSONResponse({"status": "ok"}, status_code=200)

	def available_inference(self):
		return JSONResponse(
			{"inference_map": list(self.inference_map.keys())},
			status_code=200,
		)

	async def count_tokens(self, request: CountTokenRequest):
		try:
			conv = request.conversation
			model = request.model
			return JSONResponse(
				{"ntokens": await self.inference_map[model].count_tokens(conv)},
				status_code=200,
			)
		except Exception as e:
			return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))

	def patch_endpoints(self):
		"""Register all endpoints with the FastAPI app."""
		for endpoint in self._endpoints:
			for method in endpoint.methods:
				route_params = {
					"path": endpoint.path,
					"response_model": None,
				}
				if endpoint.summary:
					route_params["summary"] = endpoint.summary
				if endpoint.tags:
					route_params["tags"] = endpoint.tags
				if method == "GET":
					self.app.get(**route_params)(endpoint.handler)
				elif method == "POST":
					self.app.post(**route_params)(endpoint.handler)

	def fire(
		self,
		host="0.0.0.0",
		port=7860,
		metrics_port: int = 7861,
		log_level="debug",
	):
		start_http_server(metrics_port)
		uvicorn.run(
			self.app,
			host=host,
			port=port,
			log_level=log_level,
			timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
			loop="uvloop",
		)

	async def async_fire(
		self,
		host="0.0.0.0",
		port=7860,
		metrics_port: int = 7861,
		log_level="debug",
	):
		start_http_server(metrics_port)
		config = uvicorn.Config(
			self.app,
			host=host,
			port=port,
			log_level=log_level,
			timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
			loop="uvloop",
		)
		server = uvicorn.Server(config)
		await server.serve()
