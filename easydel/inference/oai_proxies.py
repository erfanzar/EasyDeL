# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Enhanced FastAPI server that proxies requests to OpenAI API.

This module provides a proxy server that forwards requests to OpenAI's API
while adding vSurge-specific monitoring and compatibility features.
It enables seamless integration between EasyDeL inference engines and
OpenAI-compatible clients.

Classes:
    InferenceApiRouter: Main proxy server class with OpenAI API compatibility
    ServerStatus: Enum for server operational states
    ServerMetrics: Performance metrics tracking
    EndpointConfig: API endpoint configuration
    ErrorResponse: Standardized error response format

Example:
    >>> from easydel.inference import InferenceApiRouter
    >>> # Create a proxy to OpenAI API
    >>> router = InferenceApiRouter(
    ...     api_key="your-api-key",
    ...     base_url="https://api.openai.com/v1"
    ... )
    >>> router.run(host="0.0.0.0", port=8084)

    >>> # Or proxy to a local vSurge server
    >>> router = InferenceApiRouter(
    ...     base_url="http://localhost:8000/v1",
    ...     enable_function_calling=True
    ... )
    >>> router.run()
"""

from __future__ import annotations

import asyncio
import os
import time
import typing as tp
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .openai_api_modules import ChatCompletionRequest, CompletionRequest

if tp.TYPE_CHECKING:
    from pydantic import BaseModel

TIMEOUT_KEEP_ALIVE = 5.0


class ServerStatus(str, Enum):
    """Server status enumeration."""

    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ServerMetrics:
    """Server performance metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    average_tokens_per_second: float = 0.0
    uptime_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)


class EndpointConfig(BaseModel):
    """Configuration for a FastAPI endpoint."""

    path: str
    handler: tp.Callable
    methods: list[str]
    summary: str | None = None
    tags: list[str] | None = None
    response_model: tp.Any = None


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: dict[str, str]
    request_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


def create_error_response(status_code: HTTPStatus, message: str, request_id: str | None = None) -> JSONResponse:
    """Creates a standardized JSON error response."""
    error_response = ErrorResponse(error={"message": message, "type": status_code.name}, request_id=request_id)
    return JSONResponse(content=error_response.model_dump(), status_code=status_code.value)


class InferenceApiRouter:
    """Enhanced FastAPI server acting as an OpenAI API proxy.

    This server provides a complete OpenAI API-compatible interface that can
    forward requests to either OpenAI's API or a local vSurge/vInference server.
    It includes additional monitoring, health check, and function calling endpoints.

    The router automatically detects backend capabilities and provides appropriate
    fallbacks when features are not available.

    Attributes:
        client: AsyncOpenAI client for backend communication
        app: FastAPI application instance
        status: Current server status
        metrics: Performance metrics tracker
        base_url: Backend API base URL
        enable_function_calling: Whether function calling is enabled
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        enable_function_calling: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the Inference API Router with vSurge compatibility.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for the API
            organization: OpenAI organization ID
            enable_function_calling: Enable function calling support
            **kwargs: Additional arguments for AsyncOpenAI client
        """
        import openai
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            organization=organization,
            **kwargs,
        )
        self.openai_module = openai
        self.enable_function_calling = enable_function_calling
        self.status = ServerStatus.STARTING
        self.metrics = ServerMetrics()
        self.base_url = str(base_url) if base_url else None

        # Create FastAPI app
        self.app = FastAPI(
            title="EasyDeL Inference API Hub",
            description="High-performance inference server with OpenAI API compatibility",
            version="2.0.0",
        )

        # Register all endpoints
        self._register_endpoints()
        if enable_function_calling:
            self._add_function_calling_endpoints()

        self.status = ServerStatus.READY

    @property
    def _endpoints(self) -> list[EndpointConfig]:
        """Define all API endpoints matching vSurge API server."""
        return [
            EndpointConfig(
                path="/v1/chat/completions",
                handler=self.chat_completions,
                methods=["POST"],
                tags=["Chat"],
                summary="Create a chat completion",
            ),
            EndpointConfig(
                path="/v1/completions",
                handler=self.completions,
                methods=["POST"],
                tags=["Completions"],
                summary="Create a completion",
            ),
            EndpointConfig(
                path="/health",
                handler=self.health_check,
                methods=["GET"],
                tags=["Health"],
                summary="Comprehensive health check",
            ),
            EndpointConfig(
                path="/liveness",
                handler=self.liveness,
                methods=["GET"],
                tags=["Health"],
                summary="Liveness check",
            ),
            EndpointConfig(
                path="/readiness",
                handler=self.readiness,
                methods=["GET"],
                tags=["Health"],
                summary="Readiness check",
            ),
            EndpointConfig(
                path="/v1/models",
                handler=self.list_models,
                methods=["GET"],
                tags=["Models"],
                summary="List available models",
            ),
            EndpointConfig(
                path="/v1/models/{model_id}",
                handler=self.get_model,
                methods=["GET"],
                tags=["Models"],
                summary="Get model details",
            ),
            EndpointConfig(
                path="/metrics",
                handler=self.get_metrics,
                methods=["GET"],
                tags=["Monitoring"],
                summary="Get server metrics",
            ),
        ]

    def _add_function_calling_endpoints(self) -> None:
        """Add function calling specific endpoints."""
        additional_endpoints = [
            EndpointConfig(
                path="/v1/tools",
                handler=self.list_tools,
                methods=["GET"],
                tags=["Tools"],
                summary="List available tools/functions",
            ),
            EndpointConfig(
                path="/v1/tools/execute",
                handler=self.execute_tool,
                methods=["POST"],
                tags=["Tools"],
                summary="Execute a tool/function call",
            ),
        ]

        for endpoint in additional_endpoints:
            self.app.add_api_route(
                path=endpoint.path,
                endpoint=endpoint.handler,
                methods=endpoint.methods,
                summary=endpoint.summary,
                tags=endpoint.tags,
            )

    def _register_endpoints(self) -> None:
        """Register all API endpoints."""
        for endpoint in self._endpoints:
            self.app.add_api_route(
                path=endpoint.path,
                endpoint=endpoint.handler,
                methods=endpoint.methods,
                summary=endpoint.summary,
                tags=endpoint.tags,
                response_model=endpoint.response_model,
            )

    def build_oai_params_from_request(
        self,
        request: CompletionRequest,
    ) -> dict[str, float | int | str | bool | list]:
        """Build OpenAI parameters from completion request.

        Converts a CompletionRequest object into a dictionary of parameters
        suitable for the OpenAI API.

        Args:
            request: The completion request to convert

        Returns:
            Dictionary of OpenAI API parameters
        """
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
        request: ChatCompletionRequest,
    ) -> dict[str, float | int | str | bool | list]:
        """Build OpenAI parameters from chat completion request.

        Converts a ChatCompletionRequest object into a dictionary of parameters
        suitable for the OpenAI API, including function calling parameters if present.

        Args:
            request: The chat completion request to convert

        Returns:
            Dictionary of OpenAI API parameters with optional tool/function definitions
        """
        params = {
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

        if request.tools:
            params["tools"] = [tool.model_dump() for tool in request.tools]
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice
        if request.functions:
            params["functions"] = [func.model_dump() for func in request.functions]
        if request.function_call:
            params["function_call"] = request.function_call

        return params

    def process_request_params(
        self,
        openai_params: dict,
        request: ChatCompletionRequest | CompletionRequest,
    ) -> tuple[dict, BaseModel | None]:
        """Process request parameters before sending to OpenAI.

        Hook for subclasses to modify parameters or extract metadata
        before forwarding to the backend.

        Args:
            openai_params: Dictionary of OpenAI API parameters
            request: Original request object

        Returns:
            Tuple of (processed_params, optional_metadata)
        """
        return openai_params, None

    async def chat_completions(self, request: ChatCompletionRequest) -> tp.Any:
        """
        Handle chat completion requests with function calling support.
        (POST /v1/chat/completions)
        """
        request_id = getattr(request, "request_id", None)

        try:
            # Update metrics
            self.metrics.total_requests += 1

            openai_params = self.build_oai_params_from_chat_request(request)
            openai_params = {k: v for k, v in openai_params.items() if v is not None}
            openai_params, metadata = self.process_request_params(
                openai_params=openai_params,
                request=request,
            )

            if not request.stream:
                response = await self.client.chat.completions.create(**openai_params)
                self.metrics.successful_requests += 1
                return response
            else:
                return StreamingResponse(
                    self._stream_chat_completion(openai_params, metadata, request_id),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Request-ID": request_id or "",
                    },
                )

        except self.openai_module.APIError as e:
            self.metrics.failed_requests += 1
            raise HTTPException(status_code=e.status_code, detail=str(e)) from e
        except Exception as e:
            self.metrics.failed_requests += 1
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _stream_chat_completion(
        self, params: dict, metadata: dict | None, request_id: str | None = None
    ) -> tp.AsyncGenerator[bytes, None]:
        """Handle streaming chat completion responses.

        Streams Server-Sent Events (SSE) formatted responses from the backend.

        Args:
            params: OpenAI API parameters
            metadata: Optional metadata to include in stream
            request_id: Request identifier for tracking

        Yields:
            SSE-formatted bytes containing response chunks
        """
        try:
            stream = await self.client.chat.completions.create(**params)
            if metadata is not None:
                yield f"metadata: {metadata.model_dump_json(exclude_unset=True)}\n\n".encode()

            async for chunk in stream:
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n".encode()

            yield b"data: [DONE]\n\n"
            self.metrics.successful_requests += 1

        except Exception as e:
            self.metrics.failed_requests += 1
            error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)
            yield f"data: {error_response.body.decode()}\n\n".encode()

    async def completions(self, request: CompletionRequest) -> tp.Any:
        """
        Handle completion requests.
        (POST /v1/completions)
        """
        request_id = getattr(request, "request_id", None)

        try:
            # Update metrics
            self.metrics.total_requests += 1

            openai_params = self.build_oai_params_from_request(request)
            openai_params = {k: v for k, v in openai_params.items() if v is not None}
            openai_params, metadata = self.process_request_params(
                openai_params=openai_params,
                request=request,
            )

            if not request.stream:
                response = await self.client.completions.create(**openai_params)
                self.metrics.successful_requests += 1
                return response
            else:
                return StreamingResponse(
                    self._stream_completion(openai_params, metadata, request_id),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Request-ID": request_id or "",
                    },
                )

        except self.openai_module.APIError as e:
            self.metrics.failed_requests += 1
            raise HTTPException(status_code=e.status_code, detail=str(e)) from e
        except Exception as e:
            self.metrics.failed_requests += 1
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _stream_completion(
        self, params: dict, metadata: dict | None, request_id: str | None = None
    ) -> tp.AsyncGenerator[bytes, None]:
        """Handle streaming completion responses.

        Streams Server-Sent Events (SSE) formatted responses from the backend.

        Args:
            params: OpenAI API parameters
            metadata: Optional metadata to include in stream
            request_id: Request identifier for tracking

        Yields:
            SSE-formatted bytes containing response chunks
        """
        try:
            stream = await self.client.completions.create(**params)
            if metadata is not None:
                yield f"metadata: {metadata.model_dump_json(exclude_unset=True)}\n\n".encode()

            async for chunk in stream:
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n".encode()

            yield b"data: [DONE]\n\n"
            self.metrics.successful_requests += 1

        except Exception as e:
            self.metrics.failed_requests += 1
            error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)
            yield f"data: {error_response.body.decode()}\n\n".encode()

    async def health_check(self) -> JSONResponse:
        """
        Comprehensive health check.
        (GET /health)
        """
        try:
            # Try to list models to check if backend is responsive
            models = await self.client.models.list()
            model_count = len(models.data) if hasattr(models, "data") else 0

            health_status = {
                "status": self.status.value,
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self.metrics.start_time,
                "models": {
                    "count": model_count,
                    "backend_url": self.base_url or "default",
                },
                "metrics": {
                    "total_requests": self.metrics.total_requests,
                    "successful_requests": self.metrics.successful_requests,
                    "failed_requests": self.metrics.failed_requests,
                },
            }

            status_code = 200 if self.status == ServerStatus.READY else 503
            return JSONResponse(health_status, status_code=status_code)

        except Exception as e:
            return JSONResponse(
                {
                    "status": ServerStatus.ERROR.value,
                    "error": str(e),
                    "timestamp": time.time(),
                },
                status_code=503,
            )

    async def liveness(self) -> JSONResponse:
        """
        Liveness check endpoint.
        (GET /liveness)
        """
        return JSONResponse({"status": "alive"}, status_code=200)

    async def readiness(self) -> JSONResponse:
        """
        Readiness check endpoint.
        (GET /readiness)
        """
        try:
            await self.client.models.list()
            return JSONResponse({"status": "ready"}, status_code=200)
        except Exception as e:
            return JSONResponse({"status": "not ready", "error": str(e)}, status_code=503)

    async def get_metrics(self) -> JSONResponse:
        """
        Get server performance metrics.
        (GET /metrics)
        """
        self.metrics.uptime_seconds = time.time() - self.metrics.start_time

        # If backend is a vSurge server, try to get its metrics
        backend_metrics = None
        if self.base_url:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/metrics") as response:
                        if response.status == 200:
                            backend_metrics = await response.json()
            except Exception:
                # Backend doesn't support metrics or is unavailable
                pass

        metrics_data = {
            "api_router_metrics": {
                "uptime_seconds": self.metrics.uptime_seconds,
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "total_tokens_generated": self.metrics.total_tokens_generated,
                "average_tokens_per_second": round(self.metrics.average_tokens_per_second, 2),
                "status": self.status.value,
            },
        }

        if backend_metrics:
            metrics_data["backend_metrics"] = backend_metrics

        return JSONResponse(metrics_data, status_code=200)

    async def list_models(self) -> JSONResponse:
        """
        List available models with metadata.
        (GET /v1/models)
        """
        try:
            response = await self.client.models.list()
            models_data = []

            for model in response.data:
                model_info = {
                    "id": model.id,
                    "object": "model",
                    "created": model.created,
                    "owned_by": model.owned_by,
                    "permission": [],
                    "root": model.id,
                    "parent": None,
                }

                # Add metadata if we're connected to a vSurge backend
                if self.base_url:
                    model_info["metadata"] = {
                        "supports_chat": True,  # Assume true for now
                        "supports_function_calling": self.enable_function_calling,
                        "backend_type": "vsurge" if "vsurge" in self.base_url.lower() else "openai",
                    }

                models_data.append(model_info)

            return JSONResponse(
                {
                    "object": "list",
                    "data": models_data,
                    "total": len(models_data),
                },
                status_code=200,
            )
        except Exception as e:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    async def get_model(self, model_id: str) -> JSONResponse:
        """
        Get detailed information about a specific model.
        (GET /v1/models/{model_id})
        """
        try:
            model = await self.client.models.retrieve(model_id)

            model_info = {
                "id": model.id,
                "object": "model",
                "created": model.created,
                "owned_by": model.owned_by,
                "permission": [],
                "root": model_id,
                "parent": None,
            }

            # Add metadata
            if self.base_url:
                model_info["metadata"] = {
                    "supports_chat": True,
                    "supports_function_calling": self.enable_function_calling,
                    "backend_type": "vsurge" if "vsurge" in self.base_url.lower() else "openai",
                }

            return JSONResponse(model_info, status_code=200)

        except self.openai_module.NotFoundError:
            return create_error_response(HTTPStatus.NOT_FOUND, f"Model '{model_id}' not found")
        except Exception as e:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    async def list_tools(self) -> JSONResponse:
        """
        List available tools/functions for each model.
        (GET /v1/tools)
        """
        if not self.enable_function_calling:
            return JSONResponse(
                {
                    "message": "Function calling is disabled",
                    "tools": [],
                },
                status_code=200,
            )

        # If backend is a vSurge server, try to get its tools
        if self.base_url:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/v1/tools") as response:
                        if response.status == 200:
                            return JSONResponse(await response.json(), status_code=200)
            except Exception:
                # Backend doesn't support tools endpoint
                pass

        # Return default tools structure
        return JSONResponse(
            {
                "models": {},
                "default_format": "openai",
                "message": "Backend does not provide tools information",
            },
            status_code=200,
        )

    async def execute_tool(self, request: Request) -> JSONResponse:
        """
        Execute a tool/function call.
        (POST /v1/tools/execute)
        """
        if not self.enable_function_calling:
            return create_error_response(HTTPStatus.NOT_IMPLEMENTED, "Function calling is disabled")

        # If backend is a vSurge server, proxy the request
        if self.base_url:
            try:
                import aiohttp

                body = await request.json()

                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.base_url}/v1/tools/execute", json=body) as response:
                        response_data = await response.json()
                        return JSONResponse(response_data, status_code=response.status)
            except Exception as e:
                return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to execute tool: {e!s}")

        return create_error_response(
            HTTPStatus.NOT_IMPLEMENTED, "Tool execution endpoint is not implemented for this backend"
        )

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8084,
        log_level: str = "info",
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
        workers: int = 1,
        reload: bool = False,
    ) -> None:
        """
        Start the server with enhanced configuration.

        Args:
            host: Host address to bind to
            port: Port to listen on
            log_level: Logging level
            ssl_keyfile: Path to SSL key file
            ssl_certfile: Path to SSL certificate file
            workers: Number of worker processes
            reload: Enable auto-reload for development
        """
        uvicorn_config = {
            "app": self.app,
            "host": host,
            "port": port,
            "workers": workers if not reload else 1,
            "log_level": log_level,
            "timeout_keep_alive": TIMEOUT_KEEP_ALIVE,
            "reload": reload,
            "server_header": False,
            "date_header": True,
        }

        try:
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            pass

        if ssl_keyfile and ssl_certfile:
            uvicorn_config["ssl_keyfile"] = ssl_keyfile
            uvicorn_config["ssl_certfile"] = ssl_certfile

        uvicorn.run(**uvicorn_config)

    fire = run
