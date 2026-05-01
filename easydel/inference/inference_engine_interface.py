# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
import queue as queue_lib
import time
import traceback
import typing as tp
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
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
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    ConversationReference,
    DeltaMessage,
    FunctionCallFormat,
    FunctionDefinition,
    ResponsesRequest,
    ToolDefinition,
)
from .sampling_params import SamplingParams
from .stream_protocol import (
    build_responses_function_call_items,
    build_responses_message_item,
    build_responses_object,
    build_responses_output_items,
    build_responses_reasoning_item,
    coerce_stream_delta_message,
    compute_stream_delta_text,
    jsonify_tool_calls,
    responses_assistant_message_from_output_items,
    should_emit_responses_message_item,
)
from .typed_models import (
    ResponseFunctionCallItem,
    ResponseMessageItem,
    ResponseReasoningItem,
    ResponsesOutputItem,
    ResponsesResponse,
)

if tp.TYPE_CHECKING:
    from ..utils import ReturnSample

TIMEOUT_KEEP_ALIVE = 5.0
logger = get_logger("InferenceApiServer")


class ServerStatus(StrEnum):
    """Lifecycle states reported by every EasyDeL inference server.

    Used both as the value of :attr:`BaseInferenceApiServer.status` and
    as the ``status`` field on health-check responses, this enum gives
    operators and load balancers a single vocabulary for deciding
    whether to send traffic to a replica. The states form a directed
    graph driven by FastAPI lifespan and ``_graceful_shutdown``:
    ``STARTING -> READY -> SHUTTING_DOWN`` for a clean lifecycle, with
    ``BUSY`` / ``ERROR`` reachable from ``READY`` when the engine
    saturates or hits a fatal condition.

    Attributes:
        STARTING: Lifespan startup hooks are still running; the model
            and tokenizer are typically being loaded.
        READY: Server can accept requests and the engine is responsive.
        BUSY: Generation slots are exhausted; ``max_concurrent_generations``
            is throttling new requests with HTTP 503.
        ERROR: An unrecoverable subsystem error occurred; ``/health``
            returns 503 and operators should restart the replica.
        SHUTTING_DOWN: Shutdown is in progress; existing requests are
            being drained and no new traffic should be routed in.
    """

    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ServerMetrics:
    """Aggregate per-server counters surfaced through ``/metrics``.

    Mutated by the request-tracking middleware in
    :meth:`BaseInferenceApiServer._setup_middleware` (``total_requests``,
    ``successful_requests``, ``failed_requests``) and by streaming code
    paths that reconcile metrics after SSE failures. Token-throughput
    fields are populated by subclasses when they observe per-request
    completion stats. Operators normally consume this struct through
    the JSON payload returned by :meth:`BaseInferenceApiServer.get_metrics`.

    Attributes:
        total_requests: Cumulative count of HTTP requests handled by the
            ``add_request_id`` middleware (includes failed ones).
        successful_requests: Number of requests whose response status
            was ``< 400`` at the middleware boundary.
        failed_requests: Number of requests whose response status was
            ``>= 400`` or that raised an exception inside the handler.
        total_tokens_generated: Cumulative completion tokens emitted
            across all requests; intended to be incremented by subclasses.
        average_tokens_per_second: Rolling tokens-per-second average; the
            update strategy is left to subclasses.
        uptime_seconds: Seconds elapsed since :attr:`start_time`; refreshed
            on every request.
        start_time: Unix timestamp captured when the dataclass was
            instantiated; serves as the epoch for uptime computations.
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    average_tokens_per_second: float = 0.0
    uptime_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)


class EndpointConfig(BaseModel):
    """Declarative description of one FastAPI endpoint registered by the server.

    Concrete servers expose the entire HTTP surface as a list of these
    configs (see :meth:`BaseInferenceApiServer._endpoints`); the
    registration helper iterates the list and forwards each field to
    ``FastAPI.add_api_route``. Capturing routes this way prevents drift
    between the documented surface and the actual routes installed at
    runtime, makes it easy for subclasses to add or remove endpoints,
    and keeps the OpenAPI schema and middleware in lock-step with the
    list contents.

    Attributes:
        path: URL path template (``"/v1/chat/completions"``,
            ``"/v1/models/{model_id}"``, …) registered with FastAPI.
        handler: Async callable that processes requests reaching ``path``;
            forwarded directly as the route's ``endpoint`` parameter.
        methods: HTTP verbs the endpoint accepts (``["GET"]``,
            ``["POST"]`` and so on).
        summary: Human-readable summary surfaced in the OpenAPI schema
            and the ``/docs`` UI.
        tags: Optional documentation tags that group related endpoints
            in the OpenAPI page.
        response_model: Optional Pydantic model used by FastAPI for
            response validation and schema generation; ``None`` disables
            automatic validation.
    """

    path: str
    handler: tp.Callable
    methods: list[str]
    summary: str | None = None
    tags: list[str] | None = None
    response_model: tp.Any = None


class ErrorResponse(BaseModel):
    """Standardized JSON error envelope returned by every server endpoint.

    Used by :func:`create_error_response` to wrap HTTP-level error
    payloads so that clients see a consistent shape regardless of the
    underlying failure (validation, auth, engine 5xx, etc.). The
    structure mirrors the OpenAI error schema, with ``error.message``
    carrying the human-readable text and ``error.type`` carrying the
    HTTP status name.

    Attributes:
        error: Mapping with ``"message"`` (human-readable failure text)
            and ``"type"`` (HTTP status name like ``"BAD_REQUEST"``).
        request_id: Optional correlation ID propagated from
            ``X-Request-ID`` so operators can trace failures across
            logs and dashboards.
        timestamp: Unix timestamp captured when the response was
            constructed; defaults to the current time so middleware can
            record server-side latency without re-running ``time.time()``.
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
    """Abstract FastAPI scaffolding shared by every EasyDeL inference server.

    This base class provides the OpenAI-compatible HTTP surface — chat
    completions, completions, the new Responses API, model listing,
    metrics, health checks, and tool execution endpoints — together with
    the cross-cutting concerns that live around them (CORS, request
    metrics, stream-to-async bridging, generation-slot saturation
    control, response-store LRU caches, and SSE serialization helpers).

    Concrete servers (``eSurgeApiServer`` today, hypothetically others
    tomorrow) only need to implement the abstract methods that decide
    *how* tokens are produced (``chat_completions``, ``completions``,
    ``health_check``, ``list_models``, ``get_model``, ``get_metrics``,
    ``list_tools``, ``execute_tool``) plus the two helper hooks
    ``_create_sampling_params`` and ``_count_tokens``. Everything else —
    routing, lifecycle, shutdown draining, conversation persistence,
    Responses-payload conversion, etc. — is provided here.

    Two extension points worth knowing about:

    1. :meth:`on_startup` / :meth:`on_shutdown` are called inside the
       FastAPI lifespan context and are a good place to load models,
       open connections, or persist state.
    2. The ``_endpoints`` property is the single source of truth for the
       HTTP routes registered with FastAPI; subclasses can extend it to
       add additional endpoints without rebuilding the entire app.

    Attributes:
        thread_pool: ThreadPoolExecutor used to run blocking code paths
            (token counting, synchronous engine streams) off the asyncio
            event loop.
        max_request_size: Maximum allowed request body size in bytes.
        request_timeout: Per-request timeout in seconds (informational —
            actually enforced at the engine level).
        status: Current :class:`ServerStatus` enum value.
        metrics: :class:`ServerMetrics` instance accumulating request and
            token counters.
        app: The constructed :class:`FastAPI` application.
        enable_function_calling: Whether the ``/v1/tools`` and
            ``/v1/tools/execute`` endpoints are exposed.
        default_function_format: :class:`FunctionCallFormat` used when a
            request does not pin a specific format.
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
        """FastAPI lifespan context that drives server startup and shutdown.

        Calls :meth:`on_startup` before yielding control to the application and
        :meth:`on_shutdown` (after :meth:`_graceful_shutdown`) once shutdown is
        requested, while keeping :attr:`status` in sync with the lifecycle.

        Args:
            app: The FastAPI application instance owning the lifespan.

        Yields:
            None: The window during which the server accepts requests.
        """
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
            """Generate a custom OpenAPI schema with Bearer auth security scheme."""
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
        async def add_request_id(request: Request, call_next):  # pyright: ignore[reportUnusedFunction]
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
        async def track_metrics(request: Request, call_next):  # pyright: ignore[reportUnusedFunction]
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
        """Register additional endpoints used for function/tool calling.

        Adds two routes (``GET /v1/tools`` and ``POST /v1/tools/execute``) to
        the FastAPI application. Subclasses must implement
        :meth:`list_tools` and :meth:`execute_tool` for these to function.
        """
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
        """Register every endpoint declared by :attr:`_endpoints` with FastAPI.

        Iterates the property and calls ``add_api_route`` for each
        :class:`EndpointConfig`, propagating path, method, summary, tags, and
        response model so the OpenAPI schema stays consistent.
        """
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
        """Wait for active requests to drain before tearing down the server.

        Waits up to 30 seconds for in-flight requests to finish, then
        force-shuts the worker thread pool. Logs progress messages so
        operators can see how many requests remain.
        """
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
        """Extract tool/function definitions from a chat completion request.

        Resolves tool definitions from the request's tools field, handling
        both Pydantic model objects and raw dictionaries.

        Args:
            request: The chat completion request containing tool definitions.

        Returns:
            List of OpenAI-style tool dictionaries, or None if no tools are defined.
        """
        resolved_tools = []
        if request.tools is not None:
            for tool in request.tools:
                raw_tool: dict[str, tp.Any] | None = None
                if hasattr(tool, "model_dump"):
                    candidate = tool.model_dump(exclude_none=True)
                    if isinstance(candidate, dict):
                        raw_tool = candidate
                elif isinstance(tool, dict):
                    raw_tool = dict(tool)
                if raw_tool is None:
                    continue

                function_payload = raw_tool.get("function")
                if isinstance(function_payload, dict):
                    resolved_tools.append(
                        {"type": str(raw_tool.get("type") or "function"), "function": function_payload}
                    )
                    continue

                if isinstance(raw_tool.get("name"), str):
                    resolved_tools.append({"type": "function", "function": raw_tool})

        if len(resolved_tools) == 0 and request.functions is not None:
            for function in request.functions:
                raw_function: dict[str, tp.Any] | None = None
                if hasattr(function, "model_dump"):
                    candidate = function.model_dump(exclude_none=True)
                    if isinstance(candidate, dict):
                        raw_function = candidate
                elif isinstance(function, dict):
                    raw_function = dict(function)
                if raw_function is None or not isinstance(raw_function.get("name"), str):
                    continue
                resolved_tools.append(
                    {
                        "type": "function",
                        "function": raw_function,
                    }
                )
        if len(resolved_tools) == 0:
            return None
        return resolved_tools

    def _mark_stream_failure(self) -> None:
        """Adjust metrics when a streaming response fails after headers are sent.

        FastAPI middleware counts the request as successful as soon as a
        ``200 OK`` headers frame is emitted, so a downstream stream error must
        manually decrement ``successful_requests`` and increment
        ``failed_requests`` to keep counters honest.
        """

        if self.metrics.successful_requests > 0:
            self.metrics.successful_requests -= 1
        self.metrics.failed_requests += 1

    @staticmethod
    def _compute_delta_text(current_text: str, previous_text: str, fallback_delta: str) -> str:
        """Compute delta text by comparing accumulated text.

        Prevents token loss under concurrent streaming by computing delta from
        full accumulated text rather than relying on potentially incomplete
        ``delta_text`` values supplied by an inference engine.

        Args:
            current_text: Current cumulative output text from the engine.
            previous_text: Previously emitted cumulative text.
            fallback_delta: Delta provided by the engine, used when the
                cumulative comparison cannot be performed.

        Returns:
            The text segment newly produced since ``previous_text``.
        """
        return compute_stream_delta_text(current_text, previous_text, fallback_delta)

    @asynccontextmanager
    async def _acquire_generation_slot(
        self,
        *,
        endpoint: str | None = None,
        request_id: str | None = None,
        model: str | None = None,
        raw_request: Request | None = None,
        stream: bool | None = None,
    ) -> AsyncIterator[None]:
        """Acquire a generation slot or raise HTTP 503 when the server is saturated.

        When ``max_concurrent_generations`` was configured during init, this
        context manager pulls a token from the slot queue before yielding.
        If the queue is empty it logs detailed context and raises a 503.

        Args:
            endpoint: Endpoint name (for diagnostics only).
            request_id: Request identifier (for diagnostics only).
            model: Requested model name (for diagnostics only).
            raw_request: The raw FastAPI request (used to extract client info).
            stream: Whether the request is streaming (for diagnostics only).

        Yields:
            None: While the caller holds an exclusive generation slot.

        Raises:
            HTTPException: With ``503`` status when no slot is available.
        """

        queue = self._generation_slots
        if queue is None:
            yield
            return

        try:
            token = queue.get_nowait()
        except asyncio.QueueEmpty as e:
            client = getattr(raw_request, "client", None)
            client_host = getattr(client, "host", None)
            client_port = getattr(client, "port", None)
            forwarded_for = None
            headers = getattr(raw_request, "headers", None)
            if headers is not None and hasattr(headers, "get"):
                forwarded_for = headers.get("x-forwarded-for")

            available_generation_slots = queue.qsize()
            max_generation_slots = int(getattr(self, "_max_generation_slots", 0) or 0)
            active_generation_slots = (
                max(0, max_generation_slots - available_generation_slots) if max_generation_slots > 0 else None
            )
            status_value = getattr(self, "status", None)
            if status_value is not None:
                status_value = getattr(status_value, "value", status_value)

            logger.warning(
                "Rejecting request with HTTP 503 because all generation slots are busy. "
                "endpoint=%s request_id=%s model=%s stream=%s client_host=%s client_port=%s "
                "forwarded_for=%s server_status=%s active_http_requests=%s "
                "max_generation_slots=%s available_generation_slots=%s active_generation_slots=%s "
                "overload_message=%s",
                endpoint,
                request_id,
                model,
                stream,
                client_host,
                client_port,
                forwarded_for,
                status_value,
                len(getattr(self, "_active_requests", ()) or ()),
                max_generation_slots,
                available_generation_slots,
                active_generation_slots,
                self._overload_message,
            )
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
        stream_fn: tp.Callable[[], Iterator[tp.Any]],
    ) -> tp.Any:
        """Run blocking ``stream_fn`` in a worker thread and expose async ``get``.

        Bridges a synchronous generator (typical of inference engines) to the
        async world by hosting it on the server's thread pool and shuttling
        results through a thread-safe queue. The returned object provides an
        ``await get()`` method that yields ``("data", payload)`` tuples,
        ``("error", exc)`` on failure, and ``("end", None)`` when complete.

        Args:
            stream_fn: Zero-argument callable returning a synchronous iterator
                of streaming payloads.

        Returns:
            An ``_AsyncStreamQueue``-like object whose ``get`` coroutine
            yields the tagged payload tuples.
        """

        queue: queue_lib.Queue[tuple[str, tp.Any]] = queue_lib.Queue()

        class _AsyncStreamQueue:
            """Async-friendly accessor backed by the synchronous queue."""

            async def get(self) -> tuple[str, tp.Any]:
                """Block until a tagged payload is available and return it.

                Returns:
                    Tuple ``(kind, payload)`` where ``kind`` is one of
                    ``"data"``, ``"error"``, or ``"end"``.
                """
                while True:
                    try:
                        return queue.get_nowait()
                    except queue_lib.Empty:
                        await asyncio.sleep(0.001)

        def _enqueue(kind: str, payload: tp.Any) -> None:
            """Push ``(kind, payload)`` onto the cross-thread queue."""
            queue.put((kind, payload))

        def _producer() -> None:
            """Drive ``stream_fn`` in a worker thread, capturing exceptions."""
            try:
                for output in stream_fn():
                    _enqueue("data", output)
            except Exception as exc:
                exc.__stream_producer_traceback__ = traceback.format_exc()
                _enqueue("error", exc)
            finally:
                _enqueue("end", None)

        self.thread_pool.submit(_producer)
        return _AsyncStreamQueue()

    @staticmethod
    def _normalize_conversation_id(value: tp.Any) -> str | None:
        """Extract a conversation ID from request payload.

        Accepts a raw string, a :class:`ConversationReference`, or a dict with
        any of the keys ``id``/``conversation_id``/``conversation`` and
        returns the trimmed identifier or ``None`` when nothing valid is
        provided.

        Args:
            value: Conversation reference value pulled from a request body.

        Returns:
            The normalized conversation ID, or ``None`` if absent or invalid.
        """

        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, ConversationReference):
            conv_id = value.id or value.conversation_id or value.conversation
            if isinstance(conv_id, str):
                return conv_id.strip() or None
        if isinstance(value, dict):
            conv_id = value.get("id") or value.get("conversation_id") or value.get("conversation")
            if isinstance(conv_id, str):
                return conv_id.strip() or None
        return None

    @staticmethod
    def _lru_set(store: OrderedDict[str, tp.Any], key: str, value: tp.Any, max_size: int) -> None:
        """Insert ``key``/``value`` into ``store`` with LRU eviction.

        Args:
            store: Ordered dict acting as the LRU cache.
            key: Cache key to insert or refresh.
            value: Value to associate with the key.
            max_size: Maximum number of entries; ``0`` clears the store.
        """
        store[key] = value
        store.move_to_end(key)
        if max_size <= 0:
            store.clear()
            return
        while len(store) > max_size:
            store.popitem(last=False)

    @staticmethod
    def _conversation_from_messages(
        messages: list[ChatMessage],
        assistant_turn: str | ChatMessage,
    ) -> list[dict[str, tp.Any]]:
        """Create conversation items (excluding ``instructions``) for storage.

        Args:
            messages: Sequence of input chat messages from the request.
            assistant_turn: The assistant reply, either as a plain string or as
                a :class:`ChatMessage` with structured content.

        Returns:
            A serialized list of message dicts ready for persistence.
        """

        history = [message.model_dump(exclude_none=True) for message in messages]
        if isinstance(assistant_turn, ChatMessage):
            history.append(assistant_turn.model_dump(exclude_none=True))
        else:
            history.append({"role": "assistant", "content": assistant_turn})
        return history

    @staticmethod
    def _responses_reasoning_summary_requested(request: ResponsesRequest) -> bool:
        """Return True when reasoning summaries should be emitted in output items.

        Behavior is default-on for local OpenAI-mock parity goals and can be
        explicitly disabled using ``reasoning=False`` or
        ``reasoning.summary`` values like ``"none"``/``false``.

        Args:
            request: The Responses API request to inspect.

        Returns:
            ``True`` when the response should embed a reasoning summary item.
        """

        include = request.include
        if isinstance(include, list):
            for entry in include:
                if not isinstance(entry, str):
                    continue
                normalized = entry.strip().lower()
                if normalized == "reasoning" or normalized.startswith("reasoning.summary"):
                    return True

        reasoning = request.reasoning
        if isinstance(reasoning, bool):
            return reasoning
        if reasoning is not None:
            summary = reasoning.summary
            if summary is None:
                return True
            if isinstance(summary, bool):
                return summary
            if isinstance(summary, str):
                normalized = summary.strip().lower()
                return normalized not in {"", "none", "off", "disabled", "false", "0", "null"}
            return summary is not None
        return True

    @staticmethod
    def _normalize_chat_message(message: ChatMessage) -> ChatMessage:
        """Canonicalize multimodal/text content parts in a typed chat message.

        Converts Responses-style content parts (``input_text``, ``input_image``,
        ``input_video``) into the chat-completions equivalents expected by the
        rest of the stack. Non-list content is deep-copied unchanged.

        Args:
            message: Chat message whose content parts may need normalization.

        Returns:
            A new :class:`ChatMessage` with canonical content parts.
        """

        content = message.content
        if not isinstance(content, list):
            return message.model_copy(deep=True)

        normalized_parts: list[dict[str, tp.Any]] = []
        for part in content:
            if not isinstance(part, dict):
                normalized_parts.append({"type": "text", "text": str(part)})
                continue

            part_type = part.get("type")
            if part_type in {"input_text", "output_text"}:
                normalized_parts.append({"type": "text", "text": part.get("text", part.get("content", ""))})
            elif part_type == "text":
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

        return message.model_copy(update={"content": normalized_parts})

    @classmethod
    def _responses_payload_to_messages(
        cls,
        request: ResponsesRequest,
        *,
        include_instructions: bool = False,
    ) -> list[ChatMessage]:
        """Convert OpenAI Responses API payload into OpenAI-style chat messages.

        Notes:
            - ``instructions`` is treated as an ephemeral system message. By default we do
              not include it in the returned message list so it won't be persisted when
              implementing multi-turn state via ``previous_response_id``.

        Args:
            request: The Responses API request to flatten.
            include_instructions: When True, prepend the request's
                ``instructions`` as a system message.

        Returns:
            List of :class:`ChatMessage` objects representing the conversation
            in chat-completions form.
        """

        messages: list[ChatMessage] = []

        if include_instructions:
            instructions = request.instructions
            if isinstance(instructions, str) and instructions.strip():
                messages.append(ChatMessage(role="system", content=instructions.strip()))

        if request.messages:
            messages.extend(cls._normalize_chat_message(message) for message in request.messages)
        else:
            input_value = request.input
            if isinstance(input_value, str):
                messages.append(ChatMessage(role="user", content=input_value))
            elif isinstance(input_value, list):
                if all(
                    (isinstance(item, ChatMessage)) or (isinstance(item, dict) and "role" in item)
                    for item in input_value
                ):
                    for item in input_value:
                        if isinstance(item, ChatMessage):
                            messages.append(cls._normalize_chat_message(item))
                        else:
                            messages.append(cls._normalize_chat_message(ChatMessage.model_validate(item)))
                elif all(isinstance(item, dict) and "type" in item for item in input_value):
                    for item in tp.cast(list[dict[str, tp.Any]], input_value):
                        item_type = str(item.get("type", "")).strip().lower()
                        if item_type == "message":
                            role = item.get("role")
                            role_value = str(role).strip() if isinstance(role, str) and role.strip() else "user"
                            messages.append(
                                cls._normalize_chat_message(
                                    ChatMessage(role=role_value, content=item.get("content", ""))
                                )
                            )
                            continue

                        if item_type == "function_call_output":
                            call_id = item.get("call_id") or item.get("tool_call_id") or item.get("id")
                            output_value = item.get("output")
                            if output_value is None:
                                output_value = item.get("content")

                            if isinstance(output_value, (dict, list)):
                                output_text = json.dumps(output_value, ensure_ascii=False)
                            elif output_value is None:
                                output_text = ""
                            else:
                                output_text = str(output_value)

                            tool_msg = ChatMessage(role="tool", content=output_text)
                            if call_id is not None:
                                tool_msg.tool_call_id = str(call_id)
                            messages.append(tool_msg)
                            continue

                        if item_type in {"input_text", "output_text", "text"}:
                            text_value = item.get("text")
                            if text_value is None:
                                text_value = item.get("content", "")
                            messages.append(ChatMessage(role="user", content=str(text_value)))
                            continue

                        messages.append(ChatMessage(role="user", content=json.dumps(item, ensure_ascii=False)))
                else:
                    messages.append(ChatMessage(role="user", content=json.dumps(input_value, ensure_ascii=False)))
            elif input_value is not None:
                messages.append(ChatMessage(role="user", content=str(input_value)))

        return messages

    @staticmethod
    def _flatten_messages_to_text(messages: list[ChatMessage]) -> list[ChatMessage]:
        """Collapse content arrays into plain text for tool parsing and templating.

        Args:
            messages: Chat messages whose content may be a list of typed parts.

        Returns:
            New list of :class:`ChatMessage` instances with plain string
            content suitable for templating engines that expect strings.
        """

        flattened: list[ChatMessage] = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type in ("text", "input_text"):
                            text_parts.append(str(part.get("text", "")))
                flattened.append(
                    msg.model_copy(update={"content": " ".join(part_text for part_text in text_parts if part_text)})
                )
            else:
                flattened.append(msg.model_copy(deep=True))
        return flattened

    @staticmethod
    def _extract_responses_tools(
        request: ResponsesRequest,
    ) -> tuple[list[ToolDefinition | FunctionDefinition] | None, list[dict[str, tp.Any]] | None]:
        """Return (raw_tools, tools_for_chat_template) from a Responses payload.

        Args:
            request: The Responses API request describing the call.

        Returns:
            A two-tuple ``(raw_tools, tools_for_template)`` where the first
            element preserves the original objects for downstream parsers and
            the second contains chat-template-compatible dicts. Both are
            ``None`` when the request did not declare tools.
        """

        raw_tools = request.tools or request.functions
        if not raw_tools:
            return None, None

        tools_for_template: list[dict[str, tp.Any]] = []
        for tool in raw_tools:
            if isinstance(tool, ToolDefinition):
                tools_for_template.append(tool.model_dump(exclude_none=True))
            elif isinstance(tool, FunctionDefinition):
                tools_for_template.append({"type": "function", "function": tool.model_dump(exclude_none=True)})
            else:
                payload = tool.model_dump(exclude_none=True)
                if isinstance(payload, dict) and isinstance(payload.get("function"), dict):
                    tools_for_template.append(payload)
                else:
                    tools_for_template.append({"type": "function", "function": payload})

        return list(raw_tools), tools_for_template or None

    def _infer_sequence_length_from_engine(self, engine: tp.Any | None = None) -> int:
        """Infer maximum sequence length from the engine or fall back to 128 tokens.

        Args:
            engine: The inference engine adapter whose ``runtime_config`` is
                consulted. ``None`` triggers the fallback.

        Returns:
            ``runtime_config.max_model_len`` when available, otherwise ``128``.
        """

        runtime_config = getattr(engine, "runtime_config", None)
        max_model_len = getattr(runtime_config, "max_model_len", None) if runtime_config is not None else None
        if max_model_len:
            try:
                return int(max_model_len)
            except (TypeError, ValueError):
                pass
        return 128

    def _parse_responses_max_tokens(self, request: ResponsesRequest, engine: tp.Any | None) -> tuple[int, int | None]:
        """Return (requested_tokens_for_auth, max_tokens_for_sampling).

        Reconciles the three possible token-budget fields on a Responses
        request (``max_output_tokens``/``max_tokens``/``max_completion_tokens``)
        and produces both an authentication-side budget and an
        engine-side ``max_tokens``.

        Args:
            request: The Responses API request being processed.
            engine: Engine adapter used to derive a sensible default when the
                request does not specify a budget.

        Returns:
            Tuple ``(requested_tokens, max_tokens)`` where ``requested_tokens``
            is always a positive integer and ``max_tokens`` is ``None`` when
            sampling should run open-ended.
        """

        raw_value = request.max_output_tokens
        if raw_value is None:
            raw_value = request.max_tokens
        if raw_value is None:
            raw_value = request.max_completion_tokens

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
    def _create_sampling_params_from_responses(request: ResponsesRequest, max_tokens: int | None) -> SamplingParams:
        """Translate a Responses API payload into SamplingParams.

        Applies clamps and defaults consistent with the OpenAI reference
        implementation (temperature in ``[0.0, 2.0]``, non-negative ``top_k``)
        so downstream code can rely on validated values.

        Args:
            request: The Responses API request being translated.
            max_tokens: Pre-computed sampling budget, possibly ``None`` for
                open-ended generation.

        Returns:
            A populated :class:`SamplingParams` ready for the engine.
        """

        temperature = 1.0 if request.temperature is None else request.temperature
        top_p = 1.0 if request.top_p is None else request.top_p

        try:
            temperature_f = float(temperature)
        except (TypeError, ValueError):
            temperature_f = 1.0
        temperature_f = max(0.0, min(temperature_f, 2.0))

        try:
            top_p_f = float(top_p)
        except (TypeError, ValueError):
            top_p_f = 1.0

        stop = request.stop
        n = 1 if request.n is None else request.n

        raw_top_k = request.top_k
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
            presence_penalty=float(request.presence_penalty or 0.0),
            frequency_penalty=float(request.frequency_penalty or 0.0),
            repetition_penalty=float(request.repetition_penalty or 1.0),
            top_k=top_k,
            min_p=float(request.min_p or 0.0),
            n=int(n or 1),
            stop=stop,
        )

    @staticmethod
    def _build_responses_reasoning_item(reasoning_text: str) -> ResponseReasoningItem:
        """Build a Responses ``reasoning`` output item from raw text.

        Args:
            reasoning_text: The reasoning summary text to embed.

        Returns:
            A :class:`ResponseReasoningItem` payload object.
        """
        return build_responses_reasoning_item(reasoning_text)

    @staticmethod
    def _build_responses_function_call_items(tool_calls: list[tp.Any] | None) -> list[ResponseFunctionCallItem]:
        """Build Responses ``function_call`` output items from tool calls.

        Args:
            tool_calls: Tool call records produced by the engine, or ``None``.

        Returns:
            List of :class:`ResponseFunctionCallItem` objects (empty when
            ``tool_calls`` is falsy).
        """
        return build_responses_function_call_items(tool_calls)

    @staticmethod
    def _build_responses_message_item(output_text: str) -> ResponseMessageItem:
        """Build a Responses ``message`` output item from generated text.

        Args:
            output_text: Assistant-visible text to wrap.

        Returns:
            A :class:`ResponseMessageItem` payload object.
        """
        return build_responses_message_item(output_text)

    @staticmethod
    def _should_emit_responses_message_item(
        output_text: str,
        tool_calls: list[tp.Any] | None = None,
    ) -> bool:
        """Return whether a Responses output should include a message item.

        Args:
            output_text: Generated text to consider for inclusion.
            tool_calls: Optional tool calls accompanying the output.

        Returns:
            ``True`` when the message item should be emitted alongside any
            tool/reasoning items.
        """
        return should_emit_responses_message_item(output_text, tool_calls)

    @classmethod
    def _build_responses_output_items(
        cls,
        *,
        output_text: str,
        tool_calls: list[tp.Any] | None = None,
        reasoning_text: str | None = None,
        include_reasoning_summary: bool = False,
    ) -> list[ResponsesOutputItem]:
        """Build the ordered list of output items for a Responses payload.

        Args:
            output_text: Assistant-visible text generated by the model.
            tool_calls: Optional tool/function call records.
            reasoning_text: Optional reasoning text emitted by the model.
            include_reasoning_summary: When True, attach the reasoning summary
                even if no other reasoning content was produced.

        Returns:
            Ordered list of :class:`ResponsesOutputItem` objects suitable for
            inclusion in the final :class:`ResponsesResponse`.
        """
        return build_responses_output_items(
            output_text=output_text,
            tool_calls=tool_calls,
            reasoning_text=reasoning_text,
            include_reasoning_summary=include_reasoning_summary,
        )

    @staticmethod
    def _responses_assistant_message_from_output_items(
        output_items: list[ResponsesOutputItem],
    ) -> ChatMessage:
        """Reduce Responses output items to an assistant :class:`ChatMessage`.

        Args:
            output_items: Output items from a Responses payload.

        Returns:
            A :class:`ChatMessage` with role ``"assistant"`` summarizing the
            output items.
        """
        return responses_assistant_message_from_output_items(output_items)

    @classmethod
    def _build_responses_object(
        cls,
        *,
        response_id: str,
        model: str,
        output_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        tool_calls: list[tp.Any] | None = None,
        reasoning_text: str | None = None,
        include_reasoning_summary: bool = False,
        output_items: list[ResponsesOutputItem] | None = None,
    ) -> ResponsesResponse:
        """Construct a complete :class:`ResponsesResponse` for the Responses API.

        Args:
            response_id: Unique response identifier (e.g. ``"resp_..."``).
            model: Model identifier echoed in the response.
            output_text: Assistant-visible text.
            prompt_tokens: Token count for the prompt.
            completion_tokens: Token count for the completion.
            tool_calls: Optional tool/function call records.
            reasoning_text: Optional reasoning summary text.
            include_reasoning_summary: Whether to include the reasoning item.
            output_items: Pre-built output items; constructed automatically
                when not provided.

        Returns:
            Fully populated :class:`ResponsesResponse`.
        """
        return build_responses_object(
            response_id=response_id,
            model=model,
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=tool_calls,
            reasoning_text=reasoning_text,
            include_reasoning_summary=include_reasoning_summary,
            output_items=output_items,
        )

    @staticmethod
    def _jsonify_tool_calls(tool_calls: tp.Any) -> list[tp.Any] | None:
        """Serialize tool call objects to JSON-compatible primitives.

        Args:
            tool_calls: Tool call records produced by the engine.

        Returns:
            JSON-friendly list of dicts, or ``None`` when the input is empty.
        """
        return jsonify_tool_calls(tool_calls)

    @classmethod
    def _coerce_stream_delta_message(
        cls,
        delta_message: tp.Any,
        *,
        fallback_text: str = "",
        default_role: str | None = None,
    ) -> DeltaMessage | None:
        """Normalize parser/engine streaming deltas into a safe DeltaMessage.

        Args:
            delta_message: Raw delta object emitted by an engine or parser.
            fallback_text: Text to use when ``delta_message`` lacks a text body.
            default_role: Role to assign when the delta omits one.

        Returns:
            A :class:`DeltaMessage` ready for SSE emission, or ``None`` when
            no usable content was produced.
        """
        return coerce_stream_delta_message(
            delta_message,
            fallback_text=fallback_text,
            default_role=default_role,
        )

    @staticmethod
    def _sse_event(event: str, payload: BaseModel | dict[str, tp.Any]) -> str:
        """Format a Server-Sent Events frame for streaming responses.

        Args:
            event: SSE ``event`` field name.
            payload: Either a Pydantic model or a JSON-serializable dict.

        Returns:
            The serialized SSE frame ending with the required blank line.
        """
        payload_json = (
            payload.model_dump_json(exclude_none=True)
            if isinstance(payload, BaseModel)
            else json.dumps(payload, separators=(",", ":"))
        )
        return f"event: {event}\ndata: {payload_json}\n\n"

    async def _response_store_get_response(self, response_id: str) -> dict[str, tp.Any] | None:
        """Look up a stored Responses record by ID.

        Args:
            response_id: Identifier returned by a previous call.

        Returns:
            The stored record dict or ``None`` if absent or storage is off.
        """
        if not self._enable_response_store:
            return None

        if self._response_store_client is not None:
            return await asyncio.to_thread(self._response_store_client.get_response, response_id)

        async with self._response_store_lock:
            return self._stored_responses.get(response_id)

    async def _response_store_put_response(self, response_id: str, record: dict[str, tp.Any]) -> None:
        """Persist a Responses record under ``response_id``.

        Args:
            response_id: Identifier to store the record under.
            record: Serialized response payload to retain.
        """
        if not self._enable_response_store:
            return

        if self._response_store_client is not None:
            await asyncio.to_thread(self._response_store_client.put_response, response_id, record)
            return

        async with self._response_store_lock:
            self._lru_set(self._stored_responses, response_id, record, self._max_stored_responses)

    async def _response_store_get_conversation(self, conversation_id: str) -> list[dict[str, tp.Any]] | None:
        """Fetch the cached conversation history for a conversation ID.

        Args:
            conversation_id: Conversation identifier to look up.

        Returns:
            The list of historical messages or ``None`` when storage is off
            or no matching record exists.
        """
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
        """Persist a conversation history under ``conversation_id``.

        Args:
            conversation_id: Identifier under which the history is stored.
            history: Ordered list of messages comprising the conversation.
        """
        if not self._enable_response_store:
            return

        if self._response_store_client is not None:
            await asyncio.to_thread(self._response_store_client.put_conversation, conversation_id, history)
            return

        async with self._response_store_lock:
            self._lru_set(self._stored_conversations, conversation_id, history, self._max_stored_conversations)

    async def responses(self, request: ResponsesRequest, raw_request: Request) -> JSONResponse:
        """Handle OpenAI Responses API requests (default: not implemented).

        Subclasses should override this to implement the Responses surface.
        The base implementation returns ``501 Not Implemented``.

        Args:
            request: The parsed Responses API request body.
            raw_request: Raw FastAPI request, primarily for header access.

        Returns:
            A :class:`JSONResponse` containing an error payload.
        """
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
        """Handle ``POST /v1/chat/completions`` requests.

        Concrete subclasses must dispatch the parsed request to the
        underlying inference engine and shape the response into either a
        full :class:`ChatCompletionResponse`, an SSE :class:`StreamingResponse`
        when ``request.stream`` is set, or a :class:`JSONResponse` carrying
        an error envelope.

        Args:
            request: Parsed chat-completion request body.
            raw_request: The underlying FastAPI request, primarily used
                to read headers (``X-Request-ID``, authentication) and
                client info for diagnostics.

        Returns:
            One of :class:`ChatCompletionResponse` (non-streaming success),
            :class:`StreamingResponse` (SSE delta stream), or
            :class:`JSONResponse` (error response).
        """
        raise NotImplementedError

    @abstractmethod
    async def completions(
        self,
        request: CompletionRequest,
        raw_request: Request,
    ) -> CompletionResponse | StreamingResponse | JSONResponse:
        """Handle ``POST /v1/completions`` (legacy text-completion) requests.

        The completions endpoint operates on raw prompts (strings or
        lists of strings) without chat-template scaffolding, mirroring
        OpenAI's pre-chat completion API. Subclasses should support both
        single-shot responses and SSE streaming based on ``request.stream``.

        Args:
            request: Parsed completion request body.
            raw_request: The underlying FastAPI request used for headers
                and client info.

        Returns:
            :class:`CompletionResponse` for non-streaming requests,
            :class:`StreamingResponse` for streaming, or
            :class:`JSONResponse` carrying an error envelope.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self, raw_request: Request) -> JSONResponse:
        """Report a comprehensive health snapshot of the server.

        The response should reflect at minimum :attr:`status`, server
        uptime, active request count, and any subsystem (model, tokenizer,
        engine) state that operators or load balancers need to decide
        whether to route traffic.

        Args:
            raw_request: The raw FastAPI request (typically used only to
                propagate request IDs into the response headers).

        Returns:
            A :class:`JSONResponse` with HTTP 200 when healthy and 503
            when the server is degraded; the body must include the
            :class:`ServerStatus` value.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_metrics(self, raw_request: Request) -> JSONResponse:
        """Expose aggregate request and token-throughput counters.

        Subclasses should serialize :attr:`metrics` along with any
        engine-level counters (KV-cache utilisation, queue depth, etc.)
        in a Prometheus-friendly JSON format so SRE dashboards and
        autoscalers can consume the same payload.

        Args:
            raw_request: The raw FastAPI request, kept for symmetry with
                the other endpoints — typically unused.

        Returns:
            A :class:`JSONResponse` containing the metrics snapshot.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_models(self, raw_request: Request) -> JSONResponse:
        """Enumerate the models the server has loaded.

        The response payload should follow the OpenAI ``/v1/models``
        schema (``object: "list"`` with a ``data`` array of model
        records) so existing tooling can introspect the deployment
        without custom code paths.

        Args:
            raw_request: The raw FastAPI request used for header access.

        Returns:
            A :class:`JSONResponse` containing the model catalog.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_model(self, model_id: str, raw_request: Request) -> JSONResponse:
        """Return detailed metadata for a single loaded model.

        Subclasses should report tokenizer capabilities, context length,
        owner/version, and any feature flags (chat template, tool calling,
        reasoning) clients need to decide how to call the model.

        Args:
            model_id: The model identifier from the URL path.
            raw_request: The raw FastAPI request used for header access.

        Returns:
            A :class:`JSONResponse` with the model record, or an error
            envelope (e.g. HTTP 404) when ``model_id`` is unknown.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_tools(self, raw_request: Request) -> JSONResponse:
        """List the tools/functions registered with the server.

        Backends that bundle tools (e.g. function-calling samples or
        agent toolkits) expose them here so clients can introspect what
        is callable through ``/v1/tools/execute``.

        Args:
            raw_request: The raw FastAPI request used for header access.

        Returns:
            A :class:`JSONResponse` containing the tool catalog.
        """
        raise NotImplementedError

    @abstractmethod
    async def execute_tool(self, request: Request) -> JSONResponse:
        """Execute a tool/function call dispatched through the API.

        Subclasses route the request payload to the appropriate tool
        runner, capture its result, and return a structured response.
        Errors should map to the appropriate HTTP status codes (400 for
        bad input, 500 for runtime failures, 501 when the backend has
        no executor configured).

        Args:
            request: The raw FastAPI request whose body carries the
                tool name and arguments.

        Returns:
            A :class:`JSONResponse` containing the tool execution
            result or error envelope.
        """
        raise NotImplementedError

    # Helper methods that can be used by subclasses

    @abstractmethod
    def _create_sampling_params(self, request: ChatCompletionRequest | CompletionRequest) -> SamplingParams:
        """Translate a chat or text completion request into ``SamplingParams``.

        Concrete implementations are responsible for clamping or
        defaulting any backend-specific fields and for surfacing
        validation errors when the request asks for a configuration the
        underlying engine cannot honour.

        Args:
            request: Either a :class:`ChatCompletionRequest` or
                :class:`CompletionRequest`; both share the OpenAI-style
                generation parameters.

        Returns:
            A populated :class:`SamplingParams` ready to be handed to
            the inference engine.
        """
        raise NotImplementedError

    def _determine_finish_reason(self, tokens_generated: int, max_tokens: float, text: str) -> str:
        """Pick an OpenAI-style ``finish_reason`` from generation metadata.

        The default implementation uses the simplest budget-based rule:
        if the number of generated tokens reached the requested budget,
        the reason is ``"length"``; otherwise the model is assumed to
        have produced an end-of-sequence token and the reason is
        ``"stop"``. Subclasses can override this to surface
        ``"tool_calls"``, ``"content_filter"``, or other reasons that
        depend on engine-specific signals.

        Args:
            tokens_generated: Total tokens produced for this completion.
            max_tokens: Sampling budget the engine was asked to honour.
            text: The decoded completion text (kept for subclass hooks
                that may rely on textual signals such as forced stop
                strings).

        Returns:
            One of the literals expected by the OpenAI API
            (``"stop"``, ``"length"``, ``"tool_calls"``, …).
        """
        if tokens_generated >= max_tokens:
            return "length"
        return "stop"

    async def _count_tokens_async(self, content: str, model_name: str | None = None) -> int:
        """Run :meth:`_count_tokens` on the worker thread pool.

        Tokenization is generally CPU-bound and tokenizer libraries
        often hold the GIL during tokenize calls, so this helper
        offloads the work to :attr:`thread_pool` instead of stalling the
        asyncio event loop.

        Args:
            content: Text whose token count is needed.
            model_name: Optional model identifier passed through to
                :meth:`_count_tokens` so subclasses can dispatch to
                model-specific tokenizers.

        Returns:
            Token count produced by :meth:`_count_tokens`.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, self._count_tokens, content, model_name)

    @abstractmethod
    def _count_tokens(self, content: str, model_name: str | None = None) -> int:
        """Synchronously count tokens in ``content`` for ``model_name``.

        Implementations must use the model's own tokenizer so that the
        returned count matches what the inference engine sees on the
        prompt path, including any chat-template scaffolding.

        Args:
            content: Text whose token count is needed.
            model_name: Optional model identifier; ``None`` should fall
                back to the server's default tokenizer.

        Returns:
            Number of tokens produced by the model's tokenizer.
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
        """Launch the FastAPI server with uvicorn under the configured options.

        Spawns a uvicorn instance bound to ``host``/``port`` and (when
        available) installs ``uvloop`` as the asyncio event loop policy
        for higher throughput. SSL is enabled when both ``ssl_keyfile``
        and ``ssl_certfile`` are supplied. ``workers`` is forced to 1
        when ``reload=True`` because uvicorn's reload mode is
        incompatible with multi-worker setups.

        Args:
            host: Address to bind to (``"0.0.0.0"`` for all interfaces).
            port: TCP port to listen on.
            workers: Number of worker processes (clamped to 1 when
                ``reload`` is enabled).
            log_level: uvicorn log level (``"debug"``, ``"info"``,
                ``"warning"``, ``"error"``).
            ssl_keyfile: Optional path to an SSL private-key PEM file.
            ssl_certfile: Optional path to an SSL certificate PEM file.
            reload: Enable uvicorn's auto-reload mode (development only).
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
    """Adapter contract bridging EasyDeL's API server to an inference engine.

    This protocol decouples the FastAPI surface defined by
    :class:`BaseInferenceApiServer` from the concrete inference engine
    implementation, so the same server can be backed by eSurge, an
    external HTTP gateway, or test stubs without changes to the
    HTTP/SSE layer. Implementations wrap the engine's native API and
    expose a small set of methods that the server invokes:

    - :meth:`generate` for text generation (streaming or non-streaming),
    - :meth:`count_tokens` for prompt-length accounting,
    - :meth:`get_model_info` for the ``/v1/models`` payload,
    - the :attr:`model_name` and :attr:`processor` properties for
      identifying the active model and its tokenizer.

    The adapter is intentionally minimal so that even a thin in-memory
    test double satisfies it without dragging in the full engine.
    """

    @abstractmethod
    async def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> list[ReturnSample] | AsyncGenerator[list[ReturnSample], None]:
        """Run text generation through the underlying engine.

        Args:
            prompts: A single prompt or a batch of prompts to generate
                completions for.
            sampling_params: Sampling configuration produced by the
                server's :meth:`BaseInferenceApiServer._create_sampling_params`.
            stream: When ``True``, the adapter must return an async
                generator yielding :class:`ReturnSample` snapshots; when
                ``False``, it must return a final list.

        Returns:
            A ``list[ReturnSample]`` for non-streaming calls, or an
            ``AsyncGenerator[list[ReturnSample], None]`` for streaming
            calls (one outer list per generation step).
        """
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, content: str) -> int:
        """Return the engine-side token count for ``content``.

        The count must match what the engine itself sees during
        prompt processing so that the server can enforce context-
        length budgets accurately.

        Args:
            content: Text whose token count is needed.

        Returns:
            Tokenized length of ``content`` according to the engine's
            tokenizer.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_info(self) -> dict[str, tp.Any]:
        """Return a dictionary suitable for inclusion in ``/v1/models`` responses.

        The dict typically carries the model name, owner, capabilities
        (chat / tools / reasoning), and any engine-specific metadata
        (parameter count, quantization, attention backend) that
        operators want to surface.

        Returns:
            A model-info dict ready to be serialized as JSON.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the active model (e.g. ``"my-org/my-model"``).

        Returns:
            The model identifier echoed back in API responses.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def processor(self) -> tp.Any:
        """Return the model's tokenizer / multimodal processor.

        Used by the server for chat-template rendering, prompt token
        counting, and to share the same tokenizer with reasoning and
        tool parsers.

        Returns:
            The ``transformers``-style tokenizer or processor instance.
        """
        raise NotImplementedError
