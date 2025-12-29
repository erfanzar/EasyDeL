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
import json
import time
import typing as tp
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus

import uvicorn
from eformer.loggings import get_logger
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .openai_api_modules import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    FunctionCallFormat,
    ResponsesRequest,
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
        enable_auth_ui: bool = True,
        max_concurrent_generations: int | None = None,
        overload_message: str = "Server is busy, please try again later",
        enable_response_store: bool = True,
        default_store_responses: bool = True,
        max_stored_responses: int = 10_000,
        max_stored_conversations: int = 1_000,
        response_store_client: tp.Any | None = None,
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
            enable_auth_ui: Enable "Authorize" button in /docs for API key input
            max_concurrent_generations: Maximum concurrent inference jobs allowed. ``None`` disables the limiter.
            overload_message: Custom error message returned when all generation slots are busy.
            enable_response_store: Enable in-memory storage for /v1/responses conversation state.
            default_store_responses: Default value for the Responses API ``store`` flag when omitted.
            max_stored_responses: Maximum stored response objects (LRU evicted).
            max_stored_conversations: Maximum stored conversation histories (LRU evicted).
            response_store_client: Optional external store client (for example a ZMQ worker client).
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
        self._overload_message = overload_message

        max_slots = 0
        if max_concurrent_generations is not None:
            try:
                max_slots = int(max_concurrent_generations)
            except (TypeError, ValueError):
                max_slots = 0
        self._max_generation_slots = max(0, max_slots)
        self._generation_slots: asyncio.Queue[int] | None = None
        if self._max_generation_slots > 0:
            self._generation_slots = asyncio.Queue(self._max_generation_slots)
            for slot in range(self._max_generation_slots):
                self._generation_slots.put_nowait(slot)

        self._enable_response_store = bool(enable_response_store)
        self._default_store_responses = bool(default_store_responses)
        self._max_stored_responses = max(0, int(max_stored_responses))
        self._max_stored_conversations = max(0, int(max_stored_conversations))
        self._stored_responses: OrderedDict[str, dict[str, tp.Any]] = OrderedDict()
        self._stored_conversations: OrderedDict[str, list[dict[str, tp.Any]]] = OrderedDict()
        self._response_store_lock = asyncio.Lock()
        self._response_store_client = response_store_client

        swagger_ui_init_oauth = None
        if enable_auth_ui:
            swagger_ui_init_oauth = {
                "clientId": "swagger-ui",
                "appName": "Swagger UI",
                "usePkceWithAuthorizationCodeGrant": True,
            }

        self.app = FastAPI(
            title=server_name,
            description=server_description,
            version=server_version,
            lifespan=self._lifespan,
            swagger_ui_init_oauth=swagger_ui_init_oauth,
        )

        # Add security schemes to OpenAPI schema if auth UI is enabled
        if enable_auth_ui:
            self.app.openapi_schema = None  # Reset to regenerate
            self._configure_openapi_security()

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

    def _configure_openapi_security(self) -> None:
        """Configure OpenAPI security schemes for API key authentication.

        Adds security scheme definitions to the OpenAPI schema, which enables
        the "Authorize" button in the /docs UI. Users can enter their API key
        via Bearer token (Authorization header) or X-API-Key header.
        """

        def custom_openapi():
            if self.app.openapi_schema:
                return self.app.openapi_schema

            from fastapi.openapi.utils import get_openapi

            openapi_schema = get_openapi(
                title=self.app.title,
                version=self.app.version,
                description=self.app.description,
                routes=self.app.routes,
            )

            # Define security schemes
            openapi_schema["components"]["securitySchemes"] = {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "API Key",
                    "description": "Enter your API key as a Bearer token (e.g., `sk-...`)",
                },
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "Enter your API key in the X-API-Key header",
                },
            }

            # Apply security globally to all endpoints (optional, can be overridden per endpoint)
            # This makes the "Authorize" button appear in the UI
            openapi_schema["security"] = [
                {"BearerAuth": []},
                {"ApiKeyAuth": []},
            ]

            self.app.openapi_schema = openapi_schema
            return self.app.openapi_schema

        self.app.openapi = custom_openapi

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
        """Define all API endpoints.

        The base server exposes a predictable suite of OpenAI-compatible
        endpoints, so this list acts as the single source of truth for route
        registration. Subclasses rarely need to override individual routes;
        instead they can extend or prune the list by overriding the property and
        composing additional :class:`EndpointConfig` entries. Keeping the
        definitions centralized also makes it easier to document the surface
        area of a deployment, since API docs, middleware, and monitoring only
        have to inspect one place to understand which handlers exist.

        Each entry intentionally captures the handler callable, HTTP verbs,
        documentation metadata, and optional response model. This mirrors the
        arguments passed to ``FastAPI.add_api_route`` and prevents drift between
        declarative configuration and runtime state. By standardizing this
        schema we can build tooling (for example automated smoke tests or
        changelog generators) that iterate through the endpoints without having
        to introspect the FastAPI app directly.
        """
        return [
            EndpointConfig(
                path="/v1/responses",
                handler=self.responses,
                methods=["POST"],
                tags=["Responses"],
                summary=(
                    "Unified OpenAI Responses API endpoint. This is the modern surface that "
                    "replaces legacy completions for most clients and can represent text, "
                    "tool calls, and multimodal inputs in a single schema.\n\n"
                    "Servers may choose to implement this directly, or translate it to "
                    "`/v1/chat/completions` internally."
                ),
            ),
            EndpointConfig(
                path="/v1/chat/completions",
                handler=self.chat_completions,
                methods=["POST"],
                tags=["Chat"],
                summary=(
                    "Submit a conversation expressed as OpenAI-style chat messages and "
                    "receive assistant turns that honor streaming, function calling, and"
                    "token usage accounting. The endpoint mirrors the semantics of the "
                    "OpenAI Chat Completions API so existing SDKs and client libraries can"
                    "drop in without translation.\n\n"
                    "Use this route whenever you need multi-turn context, tool invocation,"
                    "or delta streaming—Simple text prompts should go through the plain"
                    "completions endpoint below."
                ),
            ),
            EndpointConfig(
                path="/v1/completions",
                handler=self.completions,
                methods=["POST"],
                tags=["Completions"],
                summary=(
                    "Generate text from a raw prompt without chat scaffolding. This matches"
                    "OpenAI's legacy completion API and is ideal for single-turn tasks such"
                    "as template expansion, summarization, or logit probing.\n\n"
                    "Clients receive either a full response object or a text/event-stream"
                    "when `stream=true`, making it a minimal surface for classic prompt"
                    "engineering workloads."
                ),
            ),
            EndpointConfig(
                path="/health",
                handler=self.health_check,
                methods=["GET"],
                tags=["Health"],
                summary=(
                    "Lightweight health probe that reports server status, uptime, active"
                    "request counts, and model metadata. Load balancers and orchestrators"
                    "can call this endpoint to decide whether a replica should receive"
                    "traffic.\n\n"
                    "The payload is intentionally human-readable so operators can curl the"
                    "route during incidents and immediately understand whether the server"
                    "is READY, BUSY, or encountering errors."
                ),
            ),
            EndpointConfig(
                path="/v1/models",
                handler=self.list_models,
                methods=["GET"],
                tags=["Models"],
                summary=(
                    "Enumerate every model the server has loaded along with ownership,"
                    "capabilities, and tokenizer limits. The response mirrors the OpenAI"
                    "`/v1/models` schema so existing tooling (CLI, dashboards, SDKs) can"
                    "introspect deployments without custom code.\n\n"
                    "Call this endpoint when building control planes or auditing which"
                    "models are exposed to end users."
                ),
            ),
            EndpointConfig(
                path="/v1/models/{model_id}",
                handler=self.get_model,
                methods=["GET"],
                tags=["Models"],
                summary=(
                    "Return detailed metadata for a specific model ID, including tokenizer"
                    "capabilities, architecture hints, and server ownership information."
                    "Use this to confirm feature support (e.g., chat templates or tool"
                    "calling) before dispatching a request.\n\n"
                    "The payload is stable enough to cache in control planes or config"
                    "UIs that need to display per-model characteristics."
                ),
            ),
            EndpointConfig(
                path="/metrics",
                handler=self.get_metrics,
                methods=["GET"],
                tags=["Monitoring"],
                summary=(
                    "Expose aggregated counters covering request throughput, success and"
                    "failure rates, generated tokens, and authentication statistics."
                    "Intended for SRE dashboards, autoscalers, or simple cron-based"
                    "reporting scripts.\n\n"
                    "Because it shares the same authorization story as model endpoints,"
                    "operators can lock down who may read metrics without punching holes"
                    "in infrastructure firewalls."
                ),
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

    def _mark_stream_failure(self) -> None:
        """Adjust metrics when a streaming response fails after headers are sent."""

        if self.metrics.successful_requests > 0:
            self.metrics.successful_requests -= 1
        self.metrics.failed_requests += 1

    @staticmethod
    def _compute_delta_text(current_text: str, previous_text: str, fallback_delta: str) -> str:
        """Compute delta text by comparing accumulated text.

        Prevents token loss under concurrent streaming by computing delta from
        full accumulated text rather than relying on potentially incomplete
        ``delta_text`` values supplied by an inference engine.
        """

        current_text = current_text or ""

        if current_text.startswith(previous_text):
            delta_text = current_text[len(previous_text) :]
            if not delta_text:
                delta_text = fallback_delta or ""
        else:
            if previous_text:
                logger.warning(
                    "Accumulated text doesn't start with previous text. "
                    "prev_len=%s, curr_len=%s. This may indicate state corruption or generation reset.",
                    len(previous_text),
                    len(current_text),
                )
            delta_text = fallback_delta or current_text

        if not delta_text and not previous_text:
            delta_text = current_text

        return delta_text

    @asynccontextmanager
    async def _acquire_generation_slot(self) -> tp.AsyncIterator[None]:
        """Acquire a generation slot or raise HTTP 503 when the server is saturated."""

        queue = self._generation_slots
        if queue is None:
            yield
            return

        try:
            token = queue.get_nowait()
        except asyncio.QueueEmpty as e:
            raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail=self._overload_message) from e

        try:
            yield
        finally:
            try:
                queue.put_nowait(token)
            except asyncio.QueueFull:
                logger.warning("Generation slot queue overfilled while releasing token")

    def _start_stream_task(
        self,
        stream_fn: tp.Callable[[], tp.Iterator[tp.Any]],
    ) -> asyncio.Queue[tuple[str, tp.Any]]:
        """Run blocking ``stream_fn`` in a worker thread and push results to an asyncio queue."""

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, tp.Any]] = asyncio.Queue()

        def _producer() -> None:
            try:
                for output in stream_fn():
                    asyncio.run_coroutine_threadsafe(queue.put(("data", output)), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(("error", exc)), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(("end", None)), loop).result()

        self.thread_pool.submit(_producer)
        return queue

    @staticmethod
    def _normalize_conversation_id(value: tp.Any) -> str | None:
        """Extract a conversation ID from request payload."""

        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, dict):
            conv_id = value.get("id") or value.get("conversation_id") or value.get("conversation")
            if isinstance(conv_id, str):
                return conv_id.strip() or None
        return None

    @staticmethod
    def _lru_set(store: OrderedDict[str, tp.Any], key: str, value: tp.Any, max_size: int) -> None:
        store[key] = value
        store.move_to_end(key)
        if max_size <= 0:
            store.clear()
            return
        while len(store) > max_size:
            store.popitem(last=False)

    @staticmethod
    def _conversation_from_messages(messages: list[dict[str, tp.Any]], assistant_text: str) -> list[dict[str, tp.Any]]:
        """Create conversation items (excluding ``instructions``) for storage."""

        history = list(messages)
        history.append({"role": "assistant", "content": assistant_text})
        return history

    @staticmethod
    def _responses_payload_to_messages(
        payload: dict[str, tp.Any],
        *,
        include_instructions: bool = False,
    ) -> list[dict[str, tp.Any]]:
        """Convert OpenAI Responses API payload into OpenAI-style chat messages.

        Notes:
            - ``instructions`` is treated as an ephemeral system message. By default we do
              not include it in the returned message list so it won't be persisted when
              implementing multi-turn state via ``previous_response_id``.
        """

        messages: list[dict[str, tp.Any]] = []

        if include_instructions:
            instructions = payload.get("instructions")
            if isinstance(instructions, str) and instructions.strip():
                messages.append({"role": "system", "content": instructions})

        if isinstance(payload.get("messages"), list):
            messages.extend(tp.cast(list[dict[str, tp.Any]], payload["messages"]))
        else:
            input_value = payload.get("input")
            if isinstance(input_value, str):
                messages.append({"role": "user", "content": input_value})
            elif isinstance(input_value, list):
                if all(isinstance(item, dict) and "role" in item for item in input_value):
                    messages.extend(tp.cast(list[dict[str, tp.Any]], input_value))
                else:
                    messages.append({"role": "user", "content": input_value})
            elif input_value is not None:
                messages.append({"role": "user", "content": str(input_value)})

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            normalized_parts: list[dict[str, tp.Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "input_text":
                    normalized_parts.append({"type": "text", "text": part.get("text", "")})
                elif part_type == "input_image":
                    image_url = part.get("image_url") or part.get("image") or part.get("url")
                    if image_url is not None:
                        normalized_parts.append({"type": "image_url", "image_url": image_url})
                elif part_type == "input_video":
                    video = part.get("video")
                    if video is not None:
                        normalized_parts.append({"type": "video", "video": video})
                else:
                    normalized_parts.append(part)
            msg["content"] = normalized_parts

        return messages

    @staticmethod
    def _flatten_messages_to_text(messages: list[dict[str, tp.Any]]) -> list[dict[str, tp.Any]]:
        """Collapse content arrays into plain text for tool parsing and templating."""

        flattened: list[dict[str, tp.Any]] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type in ("text", "input_text"):
                            text_parts.append(str(part.get("text", "")))
                flattened.append({**msg, "content": " ".join(part_text for part_text in text_parts if part_text)})
            else:
                flattened.append(msg)
        return flattened

    @staticmethod
    def _extract_responses_tools(payload: dict[str, tp.Any]) -> tuple[list[dict[str, tp.Any]] | None, list[dict] | None]:
        """Return (raw_tools, tools_for_chat_template) from a Responses payload."""

        raw_tools = payload.get("tools") or payload.get("functions")
        if not isinstance(raw_tools, list) or not raw_tools:
            return None, None

        tools_for_template: list[dict] = []
        for tool in raw_tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if isinstance(fn, dict):
                tools_for_template.append(fn)
            elif isinstance(tool, dict):
                tools_for_template.append(tool)

        return tp.cast(list[dict[str, tp.Any]], raw_tools), tools_for_template or None

    def _infer_sequence_length_from_engine(self, engine: tp.Any | None = None) -> int:
        """Infer maximum sequence length from the engine or fall back to 128 tokens."""

        if engine is not None and getattr(engine, "max_model_len", None):
            try:
                return int(engine.max_model_len)
            except (TypeError, ValueError):
                pass
        return 128

    def _parse_responses_max_tokens(self, payload: dict[str, tp.Any], engine: tp.Any | None) -> tuple[int, int | None]:
        """Return (requested_tokens_for_auth, max_tokens_for_sampling)."""

        raw_value = payload.get("max_output_tokens")
        if raw_value is None:
            raw_value = payload.get("max_tokens")
        if raw_value is None:
            raw_value = payload.get("max_completion_tokens")

        if raw_value is None:
            return self._infer_sequence_length_from_engine(engine), None

        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            return self._infer_sequence_length_from_engine(engine), None

        if parsed < 0:
            return self._infer_sequence_length_from_engine(engine), None

        return parsed, parsed or None

    @staticmethod
    def _create_sampling_params_from_responses(payload: dict[str, tp.Any], max_tokens: int | None) -> SamplingParams:
        """Translate a Responses API payload into SamplingParams."""

        temperature = payload.get("temperature", 1.0)
        top_p = payload.get("top_p", 1.0)

        try:
            temperature_f = float(temperature)
        except (TypeError, ValueError):
            temperature_f = 1.0
        temperature_f = max(0.0, min(temperature_f, 2.0))

        try:
            top_p_f = float(top_p)
        except (TypeError, ValueError):
            top_p_f = 1.0

        stop = payload.get("stop")
        n = payload.get("n", 1)

        raw_top_k = payload.get("top_k")
        try:
            top_k = int(raw_top_k) if raw_top_k is not None else 0
        except (TypeError, ValueError):
            top_k = 0
        if top_k < 0:
            top_k = 0

        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature_f,
            top_p=top_p_f,
            presence_penalty=float(payload.get("presence_penalty", 0.0) or 0.0),
            frequency_penalty=float(payload.get("frequency_penalty", 0.0) or 0.0),
            repetition_penalty=float(payload.get("repetition_penalty", 1.0) or 1.0),
            top_k=top_k,
            min_p=float(payload.get("min_p", 0.0) or 0.0),
            n=int(n or 1),
            stop=stop,
        )

    @staticmethod
    def _build_responses_object(
        *,
        response_id: str,
        model: str,
        output_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        tool_calls: list[tp.Any] | None = None,
    ) -> dict[str, tp.Any]:
        created_at = int(time.time())
        message: dict[str, tp.Any] = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": output_text}],
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": "completed",
            "output": [message],
            "usage": {
                "input_tokens": int(prompt_tokens),
                "output_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens) + int(completion_tokens),
            },
        }

    @staticmethod
    def _jsonify_tool_calls(tool_calls: tp.Any) -> list[tp.Any] | None:
        if not tool_calls:
            return None
        if not isinstance(tool_calls, list):
            return None
        serialized: list[tp.Any] = []
        for call in tool_calls:
            if hasattr(call, "model_dump"):
                serialized.append(call.model_dump(exclude_unset=True, exclude_none=True))
            elif isinstance(call, dict):
                serialized.append(call)
            else:
                serialized.append(call)
        return serialized

    @staticmethod
    def _sse_event(event: str, payload: dict[str, tp.Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"

    async def _response_store_get_response(self, response_id: str) -> dict[str, tp.Any] | None:
        if not self._enable_response_store:
            return None

        if self._response_store_client is not None:
            return await asyncio.to_thread(self._response_store_client.get_response, response_id)

        async with self._response_store_lock:
            return self._stored_responses.get(response_id)

    async def _response_store_put_response(self, response_id: str, record: dict[str, tp.Any]) -> None:
        if not self._enable_response_store:
            return

        if self._response_store_client is not None:
            await asyncio.to_thread(self._response_store_client.put_response, response_id, record)
            return

        async with self._response_store_lock:
            self._lru_set(self._stored_responses, response_id, record, self._max_stored_responses)

    async def _response_store_get_conversation(self, conversation_id: str) -> list[dict[str, tp.Any]] | None:
        if not self._enable_response_store:
            return None

        if self._response_store_client is not None:
            return await asyncio.to_thread(self._response_store_client.get_conversation, conversation_id)

        async with self._response_store_lock:
            return self._stored_conversations.get(conversation_id)

    async def _response_store_put_conversation(
        self,
        conversation_id: str,
        history: list[dict[str, tp.Any]],
    ) -> None:
        if not self._enable_response_store:
            return

        if self._response_store_client is not None:
            await asyncio.to_thread(self._response_store_client.put_conversation, conversation_id, history)
            return

        async with self._response_store_lock:
            self._lru_set(self._stored_conversations, conversation_id, history, self._max_stored_conversations)

    async def responses(self, request: ResponsesRequest, raw_request: Request) -> JSONResponse:
        """Handle OpenAI Responses API requests (default: not implemented)."""
        return create_error_response(
            HTTPStatus.NOT_IMPLEMENTED,
            "This server does not implement the OpenAI Responses API (/v1/responses).",
        )

    @abstractmethod
    async def chat_completions(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
        """
        Handle chat completion requests.

        Args:
            request: The chat completion request
            raw_request: Raw FastAPI request containing headers

        Returns:
            Chat completion response (streaming or non-streaming)
        """
        raise NotImplementedError

    @abstractmethod
    async def completions(
        self,
        request: CompletionRequest,
        raw_request: Request,
    ) -> CompletionResponse | StreamingResponse | JSONResponse:
        """
        Handle completion requests.

        Args:
            request: The completion request
            raw_request: Raw FastAPI request containing headers

        Returns:
            Completion response (streaming or non-streaming)
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self, raw_request: Request) -> JSONResponse:
        """
        Perform comprehensive health check.

        Args:
            raw_request: Raw FastAPI request containing headers

        Returns:
            Health status information
        """
        raise NotImplementedError

    @abstractmethod
    async def get_metrics(self, raw_request: Request) -> JSONResponse:
        """
        Get server performance metrics.

        Args:
            raw_request: Raw FastAPI request containing headers

        Returns:
            Server metrics information
        """
        raise NotImplementedError

    @abstractmethod
    async def list_models(self, raw_request: Request) -> JSONResponse:
        """
        List available models.

        Args:
            raw_request: Raw FastAPI request containing headers

        Returns:
            List of available models with metadata
        """
        raise NotImplementedError

    @abstractmethod
    async def get_model(self, model_id: str, raw_request: Request) -> JSONResponse:
        """
        Get detailed information about a specific model.

        Args:
            model_id: The model identifier
            raw_request: Raw FastAPI request containing headers

        Returns:
            Model details
        """
        raise NotImplementedError

    @abstractmethod
    async def list_tools(self, raw_request: Request) -> JSONResponse:
        """
        List available tools/functions.

        Args:
            raw_request: Raw FastAPI request containing headers

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

    This allows different inference engines (eSurge, vLLM, TGI, etc.) to be used
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
