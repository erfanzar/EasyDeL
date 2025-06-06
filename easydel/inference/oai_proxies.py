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

"""Simple FastAPI server that proxies requests to OpenAI API."""

from __future__ import annotations

import os
import typing as tp
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .openai_api_modules import ChatCompletionRequest, CompletionRequest

if tp.TYPE_CHECKING:
    from pydantic import BaseModel
else:
    BaseModel = tp.Any

TIMEOUT_KEEP_ALIVE = 5.0

APP = FastAPI(title="EasyDeL Inference API Hub")


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    """Creates a standardized JSON error response."""
    return JSONResponse({"error": {"message": message}}, status_code=status_code.value)


class InferenceApiRouter:
    """
    FastAPI server that acts as a hub for OpenAI API requests.

    This server provides endpoints mimicking the OpenAI API structure for chat completions,
    completions, and utility endpoints.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        **kwargs,
    ) -> None:
        import openai
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            organization=organization,
            **kwargs,
        )
        self.openai_module = openai

        self._register_endpoints()

    def _register_endpoints(self):
        """Registers all API endpoints with the FastAPI application."""
        endpoints = [
            ("/v1/chat/completions", self.chat_completions, ["POST"]),
            ("/v1/completions", self.completions, ["POST"]),
            ("/liveness", self.liveness, ["GET"]),
            ("/readiness", self.readiness, ["GET"]),
            ("/v1/models", self.list_models, ["GET"]),
        ]

        for path, handler, methods in endpoints:
            APP.add_api_route(path=path, endpoint=handler, methods=methods)

    def build_oai_params_from_request(
        self,
        request: CompletionRequest,
    ) -> dict[str, float | int | str | bool | list]:
        return {
            "model": request.model,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop,
            "stream": request.stream,
            "n": request.n,
        }

    def build_oai_params_from_chat_request(
        self,
        request: CompletionRequest,
    ) -> dict[str, float | int | str | bool | list]:
        return {
            "model": request.model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop,
            "stream": request.stream,
            "n": request.n,
        }

    def process_request_params(
        self,
        openai_params: dict,
        request: ChatCompletionRequest | CompletionRequest,
    ) -> tuple[dict, BaseModel | None]:
        return openai_params, None

    async def chat_completions(self, request: ChatCompletionRequest):
        """
        Handles chat completion requests (POST /v1/chat/completions).

        Forwards the request to OpenAI API and returns the response.
        """
        try:
            openai_params = self.build_oai_params_from_chat_request(request)
            openai_params = {k: v for k, v in openai_params.items() if v is not None}
            openai_params, metadata = self.process_request_params(
                openai_params=openai_params,
                request=request,
            )

            if not request.stream:
                return await self.client.chat.completions.create(**openai_params)
            else:
                return StreamingResponse(
                    self._stream_chat_completion(openai_params, metadata),
                    media_type="text/event-stream",
                )

        except self.openai_module.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=str(e)) from e
        except Exception as e:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    async def _stream_chat_completion(self, params: dict, metadata: dict | None) -> tp.AsyncGenerator[bytes, None]:
        """Handle streaming chat completion responses."""
        try:
            stream = await self.client.chat.completions.create(**params)
            if metadata is not None:
                yield f"metadata: {metadata.model_dump_json(exclude_unset=True)}\n\n".encode()
            async for chunk in stream:
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n".encode()

            yield b"data: [DONE]\n\n"

        except Exception as e:
            error_msg = f"data: {create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e)).body.decode()}\n\n"
            yield error_msg.encode("utf-8")

    async def completions(self, request: CompletionRequest):
        """
        Handles completion requests (POST /v1/completions).

        Forwards the request to OpenAI API and returns the response.
        """
        try:
            openai_params = self.build_oai_params_from_request(request)
            openai_params = {k: v for k, v in openai_params.items() if v is not None}
            openai_params, metadata = self.process_request_params(
                openai_params=openai_params,
                request=request,
            )

            if not request.stream:
                return await self.client.completions.create(**openai_params)
            else:
                return StreamingResponse(
                    self._stream_completion(openai_params, metadata),
                    media_type="text/event-stream",
                )

        except self.openai_module.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=str(e)) from e
        except Exception as e:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    async def _stream_completion(self, params: dict, metadata: dict | None) -> tp.AsyncGenerator[bytes, None]:
        """Handle streaming completion responses."""
        try:
            stream = await self.client.completions.create(**params)
            if metadata is not None:
                yield f"metadata: {metadata.model_dump_json(exclude_unset=True)}\n\n".encode()
            async for chunk in stream:
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n".encode()

            yield b"data: [DONE]\n\n"

        except Exception as e:
            error_msg = f"data: {create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e)).body.decode()}\n\n"
            yield error_msg.encode("utf-8")

    async def liveness(self):
        """Liveness check endpoint (GET /liveness)."""
        return JSONResponse({"status": "alive"}, status_code=200)

    async def readiness(self):
        """Readiness check endpoint (GET /readiness)."""
        try:
            await self.client.models.list()
            return JSONResponse({"status": "ready"}, status_code=200)
        except Exception as e:
            return JSONResponse({"status": "not ready", "error": str(e)}, status_code=503)

    async def list_models(self):
        """Lists available models (GET /v1/models)."""
        try:
            response = await self.client.models.list()
            models_data = [
                {
                    "id": model.id,
                    "object": "model",
                    "created": model.created,
                    "owned_by": model.owned_by,
                }
                for model in response.data
            ]
            return JSONResponse({"object": "list", "data": models_data}, status_code=200)
        except Exception as e:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8084,
        log_level: str = "info",
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
    ):
        """
        Starts the uvicorn server to run the FastAPI application.

        Args:
            host: The host address to bind to.
            port: The port to listen on.
            log_level: The logging level for uvicorn.
            ssl_keyfile: Path to the SSL key file for HTTPS.
            ssl_certfile: Path to the SSL certificate file for HTTPS.
        """
        uvicorn_config = {"host": host, "port": port, "log_level": log_level, "timeout_keep_alive": TIMEOUT_KEEP_ALIVE}

        try:
            import uvloop  # noqa

            uvicorn_config["loop"] = "uvloop"
        except ImportError:
            pass

        if ssl_keyfile and ssl_certfile:
            uvicorn_config["ssl_keyfile"] = ssl_keyfile
            uvicorn_config["ssl_certfile"] = ssl_certfile

        uvicorn.run(APP, **uvicorn_config)

    fire = run
