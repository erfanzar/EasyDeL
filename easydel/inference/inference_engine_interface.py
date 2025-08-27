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

"""Base interface for EasyDeL inference API servers.

This module provides abstract base classes and utilities for building
standardized inference API servers with OpenAI API compatibility.

Classes:
    ServerStatus: Enum representing server operational states
    ServerMetrics: Dataclass for tracking server performance metrics
    EndpointConfig: Configuration for API endpoints
    ErrorResponse: Standard error response format
    BaseInferenceApiServer: Abstract base class for inference servers
    InferenceEngineAdapter: Abstract adapter for different inference engines
"""

from __future__ import annotations

import asyncio
import time
import typing as tp
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus

import uvicorn
from eformer.loggings import get_logger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .openai_api_modules import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    FunctionCallFormat,
)
from .sampling_params import SamplingParams

if tp.TYPE_CHECKING:
    from ..utils import ReturnSample

TIMEOUT_KEEP_ALIVE = 5.0
logger = get_logger("InferenceApiServer")


class ServerStatus(str, Enum):
    """Server status enumeration.

    Represents the operational state of an inference server.

    Attributes:
        STARTING: Server is initializing
        READY: Server is ready to accept requests
        BUSY: Server is processing requests at capacity
        ERROR: Server encountered an error
        SHUTTING_DOWN: Server is gracefully shutting down
    """

    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ServerMetrics:
    """Server performance metrics.

    Tracks key performance indicators for the inference server.

    Attributes:
        total_requests: Total number of requests received
        successful_requests: Number of successfully completed requests
        failed_requests: Number of failed requests
        total_tokens_generated: Total tokens generated across all requests
        average_tokens_per_second: Average generation speed
        uptime_seconds: Time since server started
        start_time: Unix timestamp when server started
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    average_tokens_per_second: float = 0.0
    uptime_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)


class EndpointConfig(BaseModel):
    """Configuration for a FastAPI endpoint.

    Defines the structure for registering API endpoints.

    Attributes:
        path: URL path for the endpoint
        handler: Callable that handles requests
        methods: HTTP methods supported (GET, POST, etc.)
        summary: Brief description of the endpoint
        tags: Tags for API documentation grouping
        response_model: Pydantic model for response validation
    """

    path: str
    handler: tp.Callable
    methods: list[str]
    summary: str | None = None
    tags: list[str] | None = None
    response_model: tp.Any = None


class ErrorResponse(BaseModel):
    """Standard error response model.

    Provides a consistent error response format across all endpoints.

    Attributes:
        error: Dictionary containing error message and type
        request_id: Optional unique identifier for the request
        timestamp: Unix timestamp when error occurred
    """

    error: dict[str, str]
    request_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


def create_error_response(status_code: HTTPStatus, message: str, request_id: str | None = None) -> JSONResponse:
    """Creates a standardized JSON error response.

    Args:
        status_code: HTTP status code for the error
        message: Human-readable error message
        request_id: Optional request identifier for tracking

    Returns:
        JSONResponse with error details and appropriate status code
    """
    error_response = ErrorResponse(error={"message": message, "type": status_code.name}, request_id=request_id)
    return JSONResponse(content=error_response.model_dump(), status_code=status_code.value)


class BaseInferenceApiServer(ABC):
    """
    Abstract base class for inference API servers.

    This interface defines the standard structure and methods that all inference
    API servers should implement to ensure consistency across different inference modules.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        enable_cors: bool = True,
        cors_origins: list[str] | None = None,
        max_request_size: int = 10 * 1024 * 1024,
        request_timeout: float = 300.0,
        enable_function_calling: bool = True,
        default_function_format: FunctionCallFormat = FunctionCallFormat.OPENAI,
        server_name: str = "EasyDeL Inference API Server",
        server_description: str = "High-performance inference server with OpenAI API compatibility",
        server_version: str = "2.0.0",
    ) -> None:
        """
        Initialize the base inference API server.

        Args:
            max_workers: Maximum number of worker threads
            enable_cors: Enable CORS middleware
            cors_origins: Allowed CORS origins
            max_request_size: Maximum request size in bytes
            request_timeout: Request timeout in seconds
            enable_function_calling: Enable function calling support
            default_function_format: Default format for function calls
            server_name: Name of the server for FastAPI app
            server_description: Description of the server
            server_version: Version of the server
        """
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="inference-worker")
        self.max_request_size = max_request_size
        self.request_timeout = request_timeout
        self.status = ServerStatus.STARTING
        self.metrics = ServerMetrics()
        self._active_requests: set[str] = set()
        self._request_lock = asyncio.Lock()
        self.enable_function_calling = enable_function_calling
        self.default_function_format = default_function_format

        self.app = FastAPI(
            title=server_name,
            description=server_description,
            version=server_version,
            lifespan=self._lifespan,
        )

        if enable_cors:
            self._setup_cors(cors_origins)

        self._setup_middleware()
        self._register_endpoints()

        if enable_function_calling:
            self._add_function_calling_endpoints()

        logger.info(f"{server_name} initialized")

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Manage server lifecycle."""
        logger.info(f"Starting {self.app.title}...")
        await self.on_startup()
        self.status = ServerStatus.READY
        yield
        logger.info(f"Shutting down {self.app.title}...")
        self.status = ServerStatus.SHUTTING_DOWN
        await self._graceful_shutdown()
        await self.on_shutdown()

    async def on_startup(self) -> None:  # noqa: B027
        """Hook for server startup.

        Override in subclasses to perform custom initialization tasks
        such as loading models, establishing connections, or warming up caches.
        This method is called once when the server starts.
        """
        pass

    async def on_shutdown(self) -> None:  # noqa: B027
        """Hook for server shutdown.

        Override in subclasses to perform cleanup tasks such as
        saving state, closing connections, or releasing resources.
        This method is called once when the server shuts down.
        """
        pass

    def _setup_cors(self, origins: list[str] | None) -> None:
        """Setup CORS middleware.

        Configures Cross-Origin Resource Sharing to allow web browsers
        to make requests to the API from different domains.

        Args:
            origins: List of allowed origin URLs. Defaults to ["*"] (all origins)
        """
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_middleware(self) -> None:
        """Setup request middleware.

        Configures middleware for request tracking, metrics collection,
        and request ID generation. This method adds two middleware layers:
        1. Request ID assignment for tracking
        2. Metrics collection for monitoring
        """

        @self.app.middleware("http")
        async def add_request_id(request: Request, call_next):
            """Add unique request ID to each request."""
            request_id = f"req_{int(time.time() * 1000000)}"
            request.state.request_id = request_id

            async with self._request_lock:
                self._active_requests.add(request_id)

            try:
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response
            finally:
                async with self._request_lock:
                    self._active_requests.discard(request_id)

        @self.app.middleware("http")
        async def track_metrics(request: Request, call_next):
            """Track request metrics."""
            self.metrics.total_requests += 1

            try:
                response = await call_next(request)
                if response.status_code < 400:
                    self.metrics.successful_requests += 1
                else:
                    self.metrics.failed_requests += 1
                return response
            except Exception:
                self.metrics.failed_requests += 1
                raise
            finally:
                self.metrics.uptime_seconds = time.time() - self.metrics.start_time

    @property
    def _endpoints(self) -> list[EndpointConfig]:
        """Define all API endpoints."""
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

    async def _graceful_shutdown(self) -> None:
        """Perform graceful shutdown."""
        max_wait = 30
        start = time.time()

        while self._active_requests and (time.time() - start) < max_wait:
            logger.info(f"Waiting for {len(self._active_requests)} active requests...")
            await asyncio.sleep(1)

        if self._active_requests:
            logger.warning(f"Force closing {len(self._active_requests)} active requests")

        self.thread_pool.shutdown(wait=True)
        logger.info("Thread pool shut down")

    def extract_tools(self, request: ChatCompletionRequest) -> list[dict] | None:
        resolved_tools = []
        if request.tools is not None:
            for tool in request.tools:
                resolved_tools.append(tool.function.model_dump())
        if len(resolved_tools) == 0:
            return None
        return resolved_tools

    # Abstract methods that must be implemented by subclasses

    @abstractmethod
    async def chat_completions(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
        """
        Handle chat completion requests.

        Args:
            request: The chat completion request

        Returns:
            Chat completion response (streaming or non-streaming)
        """
        raise NotImplementedError

    @abstractmethod
    async def completions(self, request: CompletionRequest) -> CompletionResponse | StreamingResponse | JSONResponse:
        """
        Handle completion requests.

        Args:
            request: The completion request

        Returns:
            Completion response (streaming or non-streaming)
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> JSONResponse:
        """
        Perform comprehensive health check.

        Returns:
            Health status information
        """
        raise NotImplementedError

    @abstractmethod
    async def get_metrics(self) -> JSONResponse:
        """
        Get server performance metrics.

        Returns:
            Server metrics information
        """
        raise NotImplementedError

    @abstractmethod
    async def list_models(self) -> JSONResponse:
        """
        List available models.

        Returns:
            List of available models with metadata
        """
        raise NotImplementedError

    @abstractmethod
    async def get_model(self, model_id: str) -> JSONResponse:
        """
        Get detailed information about a specific model.

        Args:
            model_id: The model identifier

        Returns:
            Model details
        """
        raise NotImplementedError

    @abstractmethod
    async def list_tools(self) -> JSONResponse:
        """
        List available tools/functions.

        Returns:
            Available tools information
        """
        raise NotImplementedError

    @abstractmethod
    async def execute_tool(self, request: Request) -> JSONResponse:
        """
        Execute a tool/function call.

        Args:
            request: The tool execution request

        Returns:
            Tool execution result
        """
        raise NotImplementedError

    # Helper methods that can be used by subclasses

    @abstractmethod
    def _create_sampling_params(self, request: ChatCompletionRequest | CompletionRequest) -> SamplingParams:
        """
        Create sampling parameters from request.

        Args:
            request: The completion request

        Returns:
            Sampling parameters for the inference engine
        """
        raise NotImplementedError

    def _determine_finish_reason(self, tokens_generated: int, max_tokens: float, text: str) -> str:
        """
        Determine the finish reason for a generation.

        Args:
            tokens_generated: Number of tokens generated
            max_tokens: Maximum tokens allowed
            text: Generated text

        Returns:
            Finish reason string
        """
        if tokens_generated >= max_tokens:
            return "length"
        return "stop"

    async def _count_tokens_async(self, content: str, model_name: str | None = None) -> int:
        """
        Count tokens asynchronously.

        Args:
            content: Text content to tokenize
            model_name: Optional model name for model-specific tokenization

        Returns:
            Number of tokens
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, self._count_tokens, content, model_name)

    @abstractmethod
    def _count_tokens(self, content: str, model_name: str | None = None) -> int:
        """
        Count tokens for the given content.

        Args:
            content: Text content to tokenize
            model_name: Optional model name for model-specific tokenization

        Returns:
            Number of tokens
        """
        raise NotImplementedError

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 11556,
        workers: int = 1,
        log_level: str = "info",
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
        reload: bool = False,
    ) -> None:
        """
        Start the server with enhanced configuration.

        Args:
            host: Host address to bind to
            port: Port to listen on
            workers: Number of worker processes
            log_level: Logging level
            ssl_keyfile: Path to SSL key file
            ssl_certfile: Path to SSL certificate file
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

        if ssl_keyfile and ssl_certfile:
            uvicorn_config.update({"ssl_keyfile": ssl_keyfile, "ssl_certfile": ssl_certfile})
            logger.info(f"Starting HTTPS server on https://{host}:{port}")
        else:
            logger.info(f"Starting HTTP server on http://{host}:{port}")

        try:
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Using uvloop for enhanced performance")
        except ImportError:
            logger.info("uvloop not available, using default event loop")

        uvicorn.run(**uvicorn_config)

    fire = run


class InferenceEngineAdapter(ABC):
    """
    Abstract adapter interface for different inference engines.

    This allows different inference engines (vSurge, vLLM, TGI, etc.) to be used
    with the same API server interface.
    """

    @abstractmethod
    async def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> list[ReturnSample] | tp.AsyncGenerator[list[ReturnSample], None]:
        """
        Generate text from prompts.

        Args:
            prompts: Input prompts
            sampling_params: Sampling parameters
            stream: Whether to stream the response

        Returns:
            Generated samples (list or async generator)
        """
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, content: str) -> int:
        """
        Count tokens in the given content.

        Args:
            content: Text content

        Returns:
            Number of tokens
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_info(self) -> dict[str, tp.Any]:
        """
        Get information about the loaded model.

        Returns:
            Model information dictionary
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def processor(self) -> tp.Any:
        """Get the processor/tokenizer for the model."""
        raise NotImplementedError
