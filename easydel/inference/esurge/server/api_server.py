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

"""FastAPI server for eSurge with OpenAI API compatibility."""

from __future__ import annotations

import asyncio
import time
import traceback
import typing as tp
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus

from eformer.loggings import get_logger
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import ProcessorMixin

from easydel.workers.esurge.auth import (
    ApiKeyRole,
    EnhancedApiKeyManager,
    PermissionDenied,
    QuotaExceeded,
    RateLimitExceeded,
)

from ...inference_engine_interface import BaseInferenceApiServer, InferenceEngineAdapter
from ...openai_api_modules import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    UsageInfo,
)
from ...sampling_params import SamplingParams
from ...tools.tool_calling_mixin import ToolCallingMixin
from ..esurge_engine import RequestOutput, eSurge
from .auth_endpoints import AuthEndpointsMixin

TIMEOUT_KEEP_ALIVE = 5.0
logger = get_logger("eSurgeApiServer")

_STREAM_DATA = "data"
_STREAM_ERROR = "error"
_STREAM_END = "end"


class ServerStatus(str, Enum):
    """Server status enumeration.

    Represents the current operational state of the API server.
    Used for health checks and monitoring.

    Values:
        STARTING: Server is initializing
        READY: Server is ready to accept requests
        BUSY: Server is processing at capacity
        ERROR: Server encountered an error
        SHUTTING_DOWN: Server is shutting down gracefully
    """

    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ServerMetrics:
    """Server performance metrics.

    Tracks aggregate performance statistics for the API server.
    Updated in real-time as requests are processed.

    Attributes:
        total_requests: Total number of requests received.
        successful_requests: Number of successfully completed requests.
        failed_requests: Number of failed requests.
        total_tokens_generated: Cumulative tokens generated across all requests.
        average_tokens_per_second: Rolling average generation throughput.
        uptime_seconds: Server uptime in seconds.
        start_time: Server start timestamp.
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    average_tokens_per_second: float = 0.0
    uptime_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)


RefineSamplingParamsFn = tp.Callable[
    [SamplingParams, ChatCompletionRequest | CompletionRequest, "eSurge"],
    SamplingParams | None,
]
RefineChatRequestFn = tp.Callable[[ChatCompletionRequest], ChatCompletionRequest | None]


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: dict[str, str]
    request_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


def create_error_response(status_code: HTTPStatus, message: str, request_id: str | None = None) -> JSONResponse:
    """Creates a standardized JSON error response.

    Args:
        status_code: HTTP status code for the error.
        message: Human-readable error message.
        request_id: Optional request ID for tracking.

    Returns:
        JSONResponse with error details in OpenAI API format.
    """
    error_response = ErrorResponse(error={"message": message, "type": status_code.name}, request_id=request_id)
    return JSONResponse(content=error_response.model_dump(), status_code=status_code.value)


class eSurgeAdapter(InferenceEngineAdapter):
    """Adapter for eSurge inference engine.

    Bridges the synchronous eSurge engine with the async FastAPI server.
    Implements the InferenceEngineAdapter interface for compatibility
    with the base API server infrastructure.
    """

    def __init__(self, esurge_instance: eSurge, model_name: str):
        self.esurge = esurge_instance
        self._model_name = model_name

    async def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> list[RequestOutput] | tp.AsyncGenerator[RequestOutput, None]:
        """Generate text using eSurge engine.

        Args:
            prompts: Input prompt(s) for generation.
            sampling_params: Generation parameters.
            stream: Whether to stream results (not implemented).

        Returns:
            List of RequestOutput objects for batch generation.

        Raises:
            NotImplementedError: If stream=True (streaming not supported here).
        """
        if stream:
            raise NotImplementedError()
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.esurge.generate, prompts, sampling_params, None, False)

    def count_tokens(self, content: str) -> int:
        """Count tokens using eSurge tokenizer.

        Args:
            content: Text to tokenize.

        Returns:
            Number of tokens in the content.
        """
        return len(self.esurge.tokenizer(content)["input_ids"])

    def get_model_info(self) -> dict[str, tp.Any]:
        """Get eSurge model information.

        Returns:
            Dictionary containing model metadata: name, type, architecture,
            max_model_len, and max_num_seqs.
        """
        return {
            "name": self._model_name,
            "type": "esurge",
            "architecture": type(self.esurge.model).__name__,
            "max_model_len": self.esurge.max_model_len,
            "max_num_seqs": self.esurge.max_num_seqs,
        }

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def processor(self) -> tp.Any:
        return self.esurge.tokenizer


class eSurgeApiServer(BaseInferenceApiServer, ToolCallingMixin, AuthEndpointsMixin):
    """eSurge-specific API server implementation with OpenAI compatibility.

    Provides a FastAPI-based REST API server that exposes eSurge engines
    through OpenAI-compatible endpoints. Supports multiple models, streaming,
    function calling, and comprehensive monitoring.

    Features:
    - OpenAI API v1 compatibility (/v1/chat/completions, /v1/completions)
    - Multi-model support with dynamic routing
    - Streaming responses with Server-Sent Events (SSE)
    - Function/tool calling support
    - Real-time metrics and health monitoring
    - Thread-safe request handling
    - Production-grade authentication with RBAC, rate limiting, and audit logging
    """

    def __init__(
        self,
        esurge_map: dict[str, eSurge] | eSurge,
        oai_like_processor: bool = True,
        enable_function_calling: bool = True,
        tool_parser_name: str = "hermes",
        require_api_key: bool = False,
        admin_key: str | None = None,
        enable_audit_logging: bool = True,
        max_audit_entries: int = 10000,
        storage_dir: str | None = None,
        enable_persistence: bool = True,
        auto_save_interval: float = 60.0,
        auth_worker_client: tp.Any | None = None,
        max_concurrent_generations: int | None = None,
        overload_message: str = "Server is busy, please try again later",
        refine_sampling_params: RefineSamplingParamsFn | None = None,
        refine_chat_request: RefineChatRequestFn | None = None,
        **kwargs,
    ) -> None:
        """Initialize the eSurge API server.

        Args:
            esurge_map: Single eSurge instance or dict mapping model names to instances.
            oai_like_processor: Enable OpenAI-like processor compatibility for chat templates.
            enable_function_calling: Enable function/tool calling support.
            tool_parser_name: Name of the tool parser to use (e.g., "hermes", "qwen", etc.)
            require_api_key: Enforce API key authentication for every endpoint.
            admin_key: Optional admin key for initial setup. If provided, creates an admin key.
            enable_audit_logging: Enable comprehensive audit logging for all auth operations.
            max_audit_entries: Maximum number of audit log entries to keep in memory (default: 10000).
            storage_dir: Directory for persistent auth storage. Defaults to ~/.cache/esurge-auth/
            enable_persistence: Enable persistent storage of auth data to disk (default: True).
            auto_save_interval: Seconds between automatic saves (default: 60.0).
            auth_worker_client: Optional AuthWorkerClient instance for ZMQ-based auth (default: None, uses in-process auth).
            tool_parser_worker_client: Optional ToolParserWorkerClient instance for ZMQ-based tool parsing. If None and use_tool_parser_worker=True, spawns a worker automatically.
            use_tool_parser_worker: If True and enable_function_calling=True, automatically spawns a tool parser ZMQ worker (default: True).
            max_concurrent_generations: Maximum concurrent inference jobs allowed. Defaults to the smallest
                ``max_num_seqs`` across loaded eSurge instances when not provided.
            overload_message: Custom error message returned when all generation slots are busy.
            refine_sampling_params: Optional callable to adjust SamplingParams per request.
            refine_chat_request: Optional callable to mutate or replace chat requests before processing.
            **kwargs: Additional arguments passed to BaseInferenceApiServer.

        Raises:
            TypeError: If esurge_map values are not eSurge instances.
        """
        if isinstance(esurge_map, eSurge):
            model_name = esurge_map.esurge_name
            esurge_map = {model_name: esurge_map}

        self.esurge_map = esurge_map
        self.adapters: dict[str, eSurgeAdapter] = {}

        # Build processor map for tool parser initialization
        model_processors = {}
        for name, esurge in esurge_map.items():
            if not isinstance(esurge, eSurge):
                raise TypeError(f"Value for key '{name}' must be an instance of eSurge")
            self.adapters[name] = eSurgeAdapter(esurge, name)
            model_processors[name] = esurge.tokenizer

        # Initialize tool parsers using mixin
        self.tool_parsers = self.initialize_tool_parsers(
            model_processors=model_processors,
            tool_parser_name=tool_parser_name,
            enable_function_calling=enable_function_calling,
        )

        self.oai_like_processor = oai_like_processor
        self.metrics = ServerMetrics()
        self.status = ServerStatus.STARTING
        self._active_requests: dict[str, dict] = {}
        self.tool_parser_name = tool_parser_name
        self.model_processors = model_processors
        self._refine_sampling_params_callback = refine_sampling_params
        self._refine_chat_request_callback = refine_chat_request
        if max_concurrent_generations is not None:
            max_slots = int(max_concurrent_generations)
        else:
            inferred = [esurge.max_num_seqs for esurge in esurge_map.values() if hasattr(esurge, "max_num_seqs")]
            max_slots = min(inferred) if inferred else 0
        self._max_generation_slots = max(0, max_slots)
        self._generation_slots: asyncio.Queue[int] | None = None
        self._overload_message = overload_message

        # Initialize authentication manager (either ZMQ worker or in-process)
        self._require_api_key = bool(require_api_key)

        if auth_worker_client is not None:
            # Use ZMQ worker for authentication
            self.auth_manager = auth_worker_client
            logger.info("Using ZMQ auth worker for authentication")
        else:
            # Use in-process enhanced authentication manager
            self.auth_manager = EnhancedApiKeyManager(
                require_api_key=require_api_key,
                admin_key=admin_key,
                enable_audit_logging=enable_audit_logging,
                max_audit_entries=max_audit_entries,
                storage_dir=storage_dir,
                enable_persistence=enable_persistence,
                auto_save=enable_persistence,  # Auto-save if persistence enabled
                save_interval=auto_save_interval,
            )
            if enable_persistence:
                storage_path = storage_dir or "~/.cache/esurge-auth"
                logger.info(f"Enhanced authentication system initialized with persistent storage at: {storage_path}")
            else:
                logger.info("Enhanced authentication system initialized (in-memory only)")

        super().__init__(
            server_name="EasyDeL eSurge API Server",
            server_description="High-performance eSurge inference server with OpenAI compatibility",
            enable_function_calling=enable_function_calling,
            **kwargs,
        )

    async def on_startup(self) -> None:
        """Custom startup logic for eSurge.

        Called when the FastAPI server starts. Logs loaded models
        and sets server status to READY.
        """
        logger.info(f"Loaded {len(self.adapters)} eSurge models")
        for name in self.adapters:
            logger.info(f"  - {name}")
        if self._max_generation_slots > 0 and self._generation_slots is None:
            self._generation_slots = asyncio.Queue(self._max_generation_slots)
            for slot in range(self._max_generation_slots):
                self._generation_slots.put_nowait(slot)
        self.status = ServerStatus.READY
        logger.info("eSurge API Server is ready")

    async def on_shutdown(self) -> None:
        """Custom shutdown logic for eSurge.

        Called when the FastAPI server shuts down. Cleans up ZMQ workers.
        """
        logger.info("eSurge API Server shutting down")
        self.status = ServerStatus.SHUTTING_DOWN
        logger.info("eSurge API Server shutdown complete")

    def _get_adapter(self, model_name: str) -> eSurgeAdapter:
        """Get adapter by model name.

        Args:
            model_name: Name of the model to retrieve.

        Returns:
            eSurgeAdapter instance for the specified model.

        Raises:
            HTTPException: If model not found (404).
        """
        adapter = self.adapters.get(model_name)
        if adapter is None:
            available = list(self.adapters.keys())
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Available: {available}")
        return adapter

    def _count_tokens(self, content: str, model_name: str | None = None) -> int:
        """Count tokens for the given content.

        Args:
            content: Text to tokenize.
            model_name: Optional model to use for tokenization.

        Returns:
            Number of tokens in the content.
        """
        if model_name:
            adapter = self._get_adapter(model_name)
            return adapter.count_tokens(content)
        adapter = next(iter(self.adapters.values()))
        return adapter.count_tokens(content)

    def generate_api_key(
        self,
        name: str,
        role: tp.Any = None,
        **kwargs,
    ) -> tuple[str, tp.Any]:
        """Create and register a new random API key with enhanced features.

        Args:
            name: Human-readable name for the key.
            role: Access control role (ApiKeyRole). Defaults to USER.
            **kwargs: Additional arguments passed to auth_manager.generate_api_key()
                (description, expires_in_days, rate_limits, quota, permissions, tags, metadata)

        Returns:
            Tuple of (raw_key, metadata). Store raw_key securely - it won't be retrievable later.
        """

        if role is None:
            role = ApiKeyRole.USER
        return self.auth_manager.generate_api_key(name=name, role=role, **kwargs)

    def _extract_api_key(self, raw_request: Request) -> str | None:
        """Extract API key from Authorization header, X-API-Key header, or query param."""
        auth_header = raw_request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            return auth_header.split(" ", 1)[1].strip()

        header_key = raw_request.headers.get("X-API-Key")
        if header_key:
            return header_key.strip()

        query_key = raw_request.query_params.get("api_key")
        if query_key:
            return query_key.strip()

        query_key = raw_request.query_params.get("user")
        if query_key:
            return query_key.strip()

        return None

    @staticmethod
    def _extract_payload_api_keys(payload: tp.Any) -> list[str]:
        """Collect API key candidates embedded in the JSON payload."""
        candidates: list[str] = []

        def add_candidate(value: tp.Any) -> None:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped and stripped not in candidates:
                    candidates.append(stripped)

        extras: dict[str, tp.Any] | None = None
        if hasattr(payload, "model_extra"):
            extras = payload.model_extra or {}
        elif isinstance(payload, dict):
            extras = payload

        if isinstance(extras, dict):
            for field in ("api_key", "apiKey", "api-key", "x-api-key", "auth", "authorization", "token", "key"):
                add_candidate(extras.get(field))

        if hasattr(payload, "user"):
            add_candidate(payload.user)

        return candidates

    def _auth_system_enabled(self) -> bool:
        """Determine whether authentication enforcement is active."""

        try:
            manager = self.auth_manager
        except AttributeError:
            return False

        if manager is None:
            return False

        try:
            enabled = manager.enabled  # type: ignore[attr-defined]
        except AttributeError:
            enabled = None

        if enabled is not None:
            return bool(enabled)

        try:
            _ = manager.authorize_request  # type: ignore[attr-defined]
        except AttributeError:
            return self._require_api_key

        return True

    def _authorize_request(
        self,
        raw_request: Request,
        payload_api_keys: str | tp.Iterable[str | None] | None = None,
        endpoint: str | None = None,
        model: str | None = None,
        requested_tokens: int = 0,
    ) -> str | None:
        """Authorize request using enhanced auth system with RBAC and rate limiting.

        Args:
            raw_request: FastAPI request object.
            payload_api_keys: API keys from request payload.
            endpoint: Endpoint being accessed.
            model: Model being requested.
            requested_tokens: Number of tokens requested.

        Returns:
            Validated API key string if successful, None if auth not required.

        Raises:
            HTTPException: If authentication or authorization fails.
        """
        if not self._auth_system_enabled():
            return None

        # Collect candidate keys
        candidate_keys: list[str] = []

        def add_candidate(value: tp.Any) -> None:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped and stripped not in candidate_keys:
                    candidate_keys.append(stripped)

        if isinstance(payload_api_keys, str):
            add_candidate(payload_api_keys)
        elif payload_api_keys:
            for value in payload_api_keys:
                add_candidate(value)

        add_candidate(self._extract_api_key(raw_request))

        # Get client IP
        ip_address = raw_request.client.host if raw_request.client else None

        # Try each candidate key
        for candidate in candidate_keys:
            try:
                metadata = self.auth_manager.authorize_request(
                    raw_key=candidate,
                    ip_address=ip_address,
                    endpoint=endpoint,
                    model=model,
                    requested_tokens=requested_tokens,
                )
                # Authorization successful
                raw_request.state.api_key = candidate
                raw_request.state.api_key_metadata = metadata
                return candidate
            except (PermissionDenied, RateLimitExceeded, QuotaExceeded) as e:
                # If we have more candidates, try them. Otherwise, raise the exception.
                if candidate == candidate_keys[-1]:
                    # Last candidate failed
                    if isinstance(e, RateLimitExceeded):
                        raise HTTPException(status_code=429, detail=str(e)) from e
                    elif isinstance(e, QuotaExceeded):
                        raise HTTPException(status_code=429, detail=str(e)) from e
                    else:
                        raise HTTPException(status_code=403, detail=str(e)) from e
                continue

        # No valid key found
        try:
            require_key = self.auth_manager.require_api_key  # type: ignore[attr-defined]
        except AttributeError:
            require_key = self._require_api_key

        if require_key:
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
        return None

    def _record_api_key_usage(
        self,
        raw_request: Request | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Track per-key token usage after a request completes."""
        if raw_request is None:
            return

        api_key = getattr(raw_request.state, "api_key", None)
        if not api_key:
            return

        self.auth_manager.record_usage(api_key, prompt_tokens, completion_tokens)

    @staticmethod
    def _compute_delta_text(current_text: str, previous_text: str, fallback_delta: str) -> str:
        """Compute delta text by comparing accumulated text.

        Prevents token loss under concurrent streaming by computing delta from
        full accumulated text rather than relying on potentially incomplete
        delta_text field.

        This handles three cases:
        1. Normal incremental streaming - extract new suffix from accumulated text
        2. Empty delta despite accumulation - fallback to provided delta
        3. Text doesn't build incrementally - possible reset or state corruption

        Args:
            current_text: Current accumulated text from the output.
            previous_text: Previously accumulated text from last iteration.
            fallback_delta: Fallback delta text from output.delta_text.

        Returns:
            The computed delta text to send to the client.

        Example:
            >>> delta = _compute_delta_text(
            ...     current_text="Hello world",
            ...     previous_text="Hello",
            ...     fallback_delta=" world"
            ... )
            >>> # Returns " world" (extracted from accumulation)
        """
        current_text = current_text or ""

        # Case 1: Normal incremental streaming - extract new suffix
        if current_text.startswith(previous_text):
            delta_text = current_text[len(previous_text) :]
            if not delta_text:
                # Empty delta, fallback to output's delta (might be empty too)
                delta_text = fallback_delta or ""
        # Case 2: Text doesn't build on previous (reset/error) - use fallback
        else:
            if previous_text:
                # Log warning only if we had previous text (not first chunk)
                logger.warning(
                    f"Accumulated text doesn't start with previous text. "
                    f"prev_len={len(previous_text)}, curr_len={len(current_text)}. "
                    f"This may indicate state corruption or generation reset."
                )
            delta_text = fallback_delta or current_text

        # Case 3: First chunk - send full current text
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
        stream_fn: tp.Callable[[], tp.Iterator[RequestOutput]],
    ) -> asyncio.Queue[tuple[str, tp.Any]]:
        """Run blocking stream_fn in a worker thread and push results to an asyncio queue."""

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, tp.Any]] = asyncio.Queue()

        def _producer() -> None:
            try:
                for output in stream_fn():
                    asyncio.run_coroutine_threadsafe(queue.put((_STREAM_DATA, output)), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put((_STREAM_ERROR, exc)), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put((_STREAM_END, None)), loop).result()

        self.thread_pool.submit(_producer)
        return queue

    def _mark_stream_failure(self) -> None:
        """Adjust metrics when a streaming response fails after headers are sent."""

        if self.metrics.successful_requests > 0:
            self.metrics.successful_requests -= 1
        self.metrics.failed_requests += 1

    def _infer_sequence_length_from_engine(self, esurge: eSurge | None) -> int:
        """Infer maximum sequence length from the engine or fall back to 128 tokens."""
        if esurge is not None and getattr(esurge, "max_model_len", None):
            try:
                return int(esurge.max_model_len)
            except (TypeError, ValueError):
                pass
        if self.adapters:
            first_adapter = next(iter(self.adapters.values()), None)
            if first_adapter and getattr(first_adapter.esurge, "max_model_len", None):
                try:
                    return int(first_adapter.esurge.max_model_len)
                except (TypeError, ValueError):
                    pass
        return 128

    def _ensure_request_max_tokens(
        self,
        request: ChatCompletionRequest | CompletionRequest,
        esurge: eSurge | None = None,
    ) -> int:
        """Ensure the request has max_tokens set, inferring from the engine when missing."""
        max_tokens_raw = getattr(request, "max_tokens", None)
        if max_tokens_raw is None:
            inferred = self._infer_sequence_length_from_engine(esurge)
            try:
                request.max_tokens = inferred
            except (AttributeError, ValueError):
                logger.debug("Unable to set inferred max_tokens on request; continuing with inferred value.")
            return inferred
        return int(max_tokens_raw)

    def _prepare_sampling_params(
        self,
        request: ChatCompletionRequest | CompletionRequest,
        esurge: eSurge,
    ) -> SamplingParams:
        """Create sampling params and allow optional refinement."""
        sampling_params = self._create_sampling_params(request)
        if self._refine_sampling_params_callback:
            refined = self._refine_sampling_params_callback(sampling_params, request, esurge)
            if refined is not None:
                sampling_params = refined
        return sampling_params

    def _create_sampling_params(self, request: ChatCompletionRequest | CompletionRequest) -> SamplingParams:
        """Create sampling parameters from request.

        Converts OpenAI API request parameters to eSurge SamplingParams.
        Applies validation and defaults.

        Args:
            request: OpenAI API request object.

        Returns:
            SamplingParams configured for eSurge generation.
        """
        max_tokens = int(request.max_tokens or 128)
        temperature = max(0.0, min(float(request.temperature or 1.0), 2.0))

        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            presence_penalty=float(request.presence_penalty or 0.0),
            frequency_penalty=float(request.frequency_penalty or 0.0),
            repetition_penalty=float(getattr(request, "repetition_penalty", 1.0)),
            top_k=int(getattr(request, "top_k", 50)),
            top_p=float(request.top_p or 1.0),
            min_p=float(getattr(request, "min_p", 0.0)),
            n=int(request.n or 1),
            stop=request.stop,
        )

    def _prepare_chat_input(self, request: ChatCompletionRequest, esurge: eSurge) -> str:
        """Prepare chat input for model.

        Applies chat template to message history, handling OpenAI format
        conversion if needed.

        Args:
            request: Chat completion request with messages.
            esurge: eSurge instance with tokenizer.

        Returns:
            Formatted prompt string ready for generation.

        Raises:
            RuntimeError: If chat template application fails.
        """
        conversation = request.model_dump(exclude_unset=True)["messages"]
        processor = esurge.tokenizer

        if isinstance(processor, ProcessorMixin) and self.oai_like_processor:
            from easydel.trainers.prompt_utils import convert_to_openai_format

            conversation = convert_to_openai_format(conversation)

        for msg in conversation:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                msg["content"] = " ".join(text_parts)

        try:
            if request.chat_template_kwargs is None:
                request.chat_template_kwargs = {}
            add_generation_prompt = request.chat_template_kwargs.pop("add_generation_prompt", True)
            return processor.apply_chat_template(
                tokenize=False,
                conversation=conversation,
                add_generation_prompt=add_generation_prompt,
                tools=self.extract_tools(request=request),
                **request.chat_template_kwargs,
            )
        except Exception as e:
            logger.exception(f"Error applying chat template: {e}")
            raise RuntimeError(f"Error tokenizing input: {e}") from e

    async def _prepare_chat_input_async(self, request: ChatCompletionRequest, esurge: eSurge) -> str:
        """Prepare chat input asynchronously.

        Async wrapper for _prepare_chat_input using thread pool.

        Args:
            request: Chat completion request.
            esurge: eSurge instance.

        Returns:
            Formatted prompt string.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, self._prepare_chat_input, request, esurge)

    async def chat_completions(self, request: ChatCompletionRequest, raw_request: Request) -> tp.Any:
        """Handle chat completion requests.

        Main endpoint for /v1/chat/completions. Supports both streaming and
        non-streaming responses, with optional function calling.

        Args:
            request: Chat completion request (with or without tools).

        Returns:
            ChatCompletionResponse for non-streaming.
            StreamingResponse for streaming.
            JSONResponse with error on failure.

        Raises:
            HTTPException: For client errors (400, 404).
        """
        request_id = str(uuid.uuid4())
        if self._refine_chat_request_callback:
            refined_request = self._refine_chat_request_callback(request)
            if refined_request is not None:
                request = refined_request
        payload_api_keys = self._extract_payload_api_keys(request)

        try:
            adapter = self._get_adapter(request.model)
            esurge = adapter.esurge

            max_tokens = self._ensure_request_max_tokens(request, esurge)
            self._authorize_request(
                raw_request,
                payload_api_keys=payload_api_keys,
                endpoint="/v1/chat/completions",
                model=request.model,
                requested_tokens=max_tokens,
            )

            if not request.messages:
                raise HTTPException(400, "Messages cannot be empty")

            content = await self._prepare_chat_input_async(request, esurge)

            if request.stream:
                return await self._handle_chat_streaming(request, esurge, content, request_id, raw_request)
            else:
                return await self._handle_chat_completion(request, esurge, content, request_id, raw_request)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            logger.exception(f"Error in chat completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _handle_chat_completion(
        self,
        request: ChatCompletionRequest,
        esurge: eSurge,
        content: str,
        request_id: str,
        raw_request: Request,
    ) -> ChatCompletionResponse:
        """Handle non-streaming chat completion.

        Generates complete response and returns it as a single object.
        Handles function call parsing if enabled.

        Args:
            request: Original chat request.
            esurge: eSurge engine instance.
            content: Formatted prompt.
            request_id: Unique request ID.

        Returns:
            Complete chat response with usage statistics.

        Raises:
            RuntimeError: If generation fails.
        """
        async with self._acquire_generation_slot():
            prompt_tokens = len(esurge.tokenizer(content)["input_ids"])

            sampling_params = self._prepare_sampling_params(request, esurge)
            loop = asyncio.get_running_loop()

            def _run_generate() -> list[RequestOutput]:
                return esurge.generate(content, sampling_params, use_tqdm=False)

            outputs = await loop.run_in_executor(self.thread_pool, _run_generate)

            if not outputs:
                raise RuntimeError("Generation failed to produce output")

            output = outputs[0]

            completion_tokens = output.num_generated_tokens
            self.metrics.total_tokens_generated += completion_tokens
            tokens_per_second = output.tokens_per_second
            processing_time = output.processing_time

            if self.metrics.average_tokens_per_second == 0:
                self.metrics.average_tokens_per_second = tokens_per_second
            else:
                self.metrics.average_tokens_per_second = (
                    self.metrics.average_tokens_per_second * 0.9 + tokens_per_second * 0.1
                )

            choices = []
            for idx, completion in enumerate(output.outputs):
                response_text = output.accumulated_text

                # Use mixin method for tool extraction
                message, finish_reason_extracted = self.extract_tool_calls_batch(
                    response_text=response_text,
                    request=request,
                    model_name=request.model,
                )
                # Override finish_reason if it was set by completion
                if finish_reason_extracted != "function_call" and completion.finish_reason:
                    finish_reason = completion.finish_reason
                else:
                    finish_reason = finish_reason_extracted
                if finish_reason == "finished":
                    finish_reason = "stop"
                choices.append(ChatCompletionResponseChoice(index=idx, message=message, finish_reason=finish_reason))

            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                tokens_per_second=tokens_per_second,
                processing_time=processing_time,
                first_token_time=output.first_token_time,
            )

            self._record_api_key_usage(raw_request, prompt_tokens, completion_tokens)

            return ChatCompletionResponse(
                model=request.model,
                choices=choices,
                usage=usage,
            )

    async def _handle_chat_streaming(
        self,
        request: ChatCompletionRequest,
        esurge: eSurge,
        content: str,
        request_id: str,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming chat completion with delta chunks.

        Streams incremental text as Server-Sent Events. Uses delta_text
        field for efficient streaming of only new content.

        Args:
            request: Original chat request.
            esurge: eSurge engine instance.
            content: Formatted prompt.
            request_id: Unique request ID.

        Returns:
            StreamingResponse with SSE format:
            - Initial role chunk
            - Content delta chunks as generated
            - Final chunk with finish_reason
            - [DONE] marker

        The streaming format follows OpenAI's SSE specification with
        'data: {...}' lines for each chunk.
        """

        sampling_params = self._prepare_sampling_params(request, esurge)

        async def generate_stream():
            async with self._acquire_generation_slot():
                prompt_tokens = len(esurge.tokenizer(content)["input_ids"])
                tool_parser = self.get_tool_parser_for_model(request.model)
                previous_text = ""
                previous_token_ids: list[int] = []
                queue = self._start_stream_task(lambda: esurge.stream(content, sampling_params))
                total_generated = 0
                generation_time = 0.0
                tokens_per_second = 0.0
                last_output: RequestOutput | None = None

                try:
                    while True:
                        kind, payload = await queue.get()
                        if kind == _STREAM_END:
                            break
                        if kind == _STREAM_ERROR:
                            raise tp.cast(Exception, payload)

                        output = tp.cast(RequestOutput, payload)
                        last_output = output
                        current_completion_tokens = output.num_generated_tokens
                        current_tps = output.tokens_per_second
                        elapsed_time = output.processing_time

                        current_text = output.accumulated_text or ""
                        delta_text = output.delta_text

                        if tool_parser:
                            current_token_ids = output.outputs[0].token_ids if output.outputs else []
                            delta_token_ids = (
                                current_token_ids[len(previous_token_ids) :] if previous_token_ids else current_token_ids
                            )

                            delta_message = self.extract_tool_calls_streaming(
                                model_name=request.model,
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request,
                            )
                            previous_text = current_text
                            previous_token_ids = current_token_ids

                            if delta_message:
                                if not delta_message.role:
                                    delta_message.role = "assistant"
                            elif request.tools:
                                continue
                            else:
                                delta_message = DeltaMessage(content=delta_text, role="assistant")
                        else:
                            previous_text = current_text
                            current_token_ids = output.outputs[0].token_ids if output.outputs else []
                            previous_token_ids = current_token_ids
                            delta_message = DeltaMessage(content=delta_text, role="assistant")

                        chunk = ChatCompletionStreamResponse(
                            model=request.model,
                            choices=[
                                ChatCompletionStreamResponseChoice(
                                    index=0,
                                    delta=delta_message,
                                    finish_reason=None,
                                )
                            ],
                            usage=UsageInfo(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=current_completion_tokens,
                                total_tokens=prompt_tokens + current_completion_tokens,
                                tokens_per_second=current_tps,
                                processing_time=elapsed_time,
                                first_token_time=output.first_token_time,
                            ),
                        )
                        yield f"data: {chunk.model_dump_json(exclude_unset=True, exclude_none=True)}\n\n"
                        total_generated = output.num_generated_tokens
                        generation_time = output.processing_time
                        tokens_per_second = output.tokens_per_second

                    if last_output is None:
                        raise RuntimeError("Streaming finished without any output")

                    usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=total_generated,
                        total_tokens=prompt_tokens + total_generated,
                        tokens_per_second=tokens_per_second,
                        processing_time=generation_time,
                        first_token_time=last_output.first_token_time,
                    )

                    final_chunk = ChatCompletionStreamResponse(
                        model=request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(content="", role="assistant"),
                                finish_reason="stop",
                            )
                        ],
                        usage=usage,
                    )

                    yield f"data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n"
                    yield "data: [DONE]\n\n"

                    self.metrics.total_tokens_generated += total_generated
                    self._record_api_key_usage(raw_request, prompt_tokens, total_generated)

                except Exception as e:
                    self._mark_stream_failure()
                    logger.exception(f"Error during streaming: {e}")
                    error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)
                    yield f"data: {error_response.body.decode()}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )

    async def completions(self, request: CompletionRequest, raw_request: Request) -> tp.Any:
        """Handle completion requests.

        Endpoint for /v1/completions. Simpler text completion without
        chat formatting.

        Args:
            request: Completion request.

        Returns:
            CompletionResponse or StreamingResponse.
            JSONResponse with error on failure.

        Raises:
            HTTPException: For client errors.
        """
        request_id = str(uuid.uuid4())
        payload_api_keys = self._extract_payload_api_keys(request)

        try:
            adapter = self._get_adapter(request.model)
            esurge = adapter.esurge

            max_tokens = self._ensure_request_max_tokens(request, esurge)
            self._authorize_request(
                raw_request,
                payload_api_keys=payload_api_keys,
                endpoint="/v1/completions",
                model=request.model,
                requested_tokens=max_tokens,
            )

            prompt = request.prompt
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""

            if not prompt:
                raise HTTPException(400, "Prompt cannot be empty")

            if request.stream:
                return await self._handle_completion_streaming(request, esurge, prompt, request_id, raw_request)
            else:
                return await self._handle_completion_response(request, esurge, prompt, request_id, raw_request)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _handle_completion_response(
        self,
        request: CompletionRequest,
        esurge: eSurge,
        prompt: str,
        request_id: str,
        raw_request: Request,
    ) -> CompletionResponse:
        """Handle non-streaming completion.

        Args:
            request: Original completion request.
            esurge: eSurge engine instance.
            prompt: Text prompt.
            request_id: Unique request ID.

        Returns:
            Complete response with generated text and usage.

        Raises:
            RuntimeError: If generation fails.
        """
        async with self._acquire_generation_slot():
            prompt_tokens = len(esurge.tokenizer(prompt)["input_ids"])
            sampling_params = self._prepare_sampling_params(request, esurge)
            loop = asyncio.get_running_loop()

            def _run_generate() -> list[RequestOutput]:
                return esurge.generate(prompt, sampling_params, use_tqdm=False)

            outputs = await loop.run_in_executor(self.thread_pool, _run_generate)

            if not outputs:
                raise RuntimeError("Generation failed to produce output")

            output = outputs[0]

            completion_tokens = output.num_generated_tokens
            self.metrics.total_tokens_generated += completion_tokens
            tokens_per_second = output.tokens_per_second
            processing_time = output.processing_time

            choices = []
            for idx, completion in enumerate(output.outputs):
                choices.append(
                    CompletionResponseChoice(
                        index=idx,
                        text=output.accumulated_text,
                        finish_reason=completion.finish_reason or "stop",
                    )
                )

            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                tokens_per_second=tokens_per_second,
                processing_time=processing_time,
                first_token_time=output.first_token_time,
            )

            self._record_api_key_usage(raw_request, prompt_tokens, completion_tokens)

            return CompletionResponse(
                model=request.model,
                choices=choices,
                usage=usage,
            )

    async def _handle_completion_streaming(
        self,
        request: CompletionRequest,
        esurge: eSurge,
        prompt: str,
        request_id: str,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming completion with delta chunks.

        Similar to chat streaming but for raw text completion.

        Args:
            request: Original completion request.
            esurge: eSurge engine instance.
            prompt: Text prompt.
            request_id: Unique request ID.

        Returns:
            StreamingResponse with incremental text chunks.
        """

        sampling_params = self._prepare_sampling_params(request, esurge)

        async def generate_stream():
            async with self._acquire_generation_slot():
                prompt_tokens = len(esurge.tokenizer(prompt)["input_ids"])
                previous_text = ""
                queue = self._start_stream_task(lambda: esurge.stream(prompt, sampling_params))
                total_generated = 0
                generation_time = 0.0
                tokens_per_second = 0.0
                last_output: RequestOutput | None = None

                try:
                    while True:
                        kind, payload = await queue.get()
                        if kind == _STREAM_END:
                            break
                        if kind == _STREAM_ERROR:
                            raise tp.cast(Exception, payload)

                        output = tp.cast(RequestOutput, payload)
                        last_output = output
                        current_completion_tokens = output.num_generated_tokens
                        current_tps = output.tokens_per_second
                        elapsed_time = output.processing_time

                        current_text = output.accumulated_text or ""
                        delta_text = self._compute_delta_text(current_text, previous_text, output.delta_text or "")
                        previous_text = current_text

                        chunk = ChatCompletionStreamResponse(
                            model=request.model,
                            choices=[
                                ChatCompletionStreamResponseChoice(
                                    index=0,
                                    delta=DeltaMessage(content=delta_text, role="assistant"),
                                    finish_reason=None,
                                )
                            ],
                            usage=UsageInfo(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=current_completion_tokens,
                                total_tokens=prompt_tokens + current_completion_tokens,
                                tokens_per_second=current_tps,
                                processing_time=elapsed_time,
                                first_token_time=output.first_token_time,
                            ),
                        )
                        yield f"data: {chunk.model_dump_json(exclude_unset=True, exclude_none=True)}\n\n"
                        total_generated = output.num_generated_tokens
                        generation_time = output.processing_time
                        tokens_per_second = output.tokens_per_second

                    if last_output is None:
                        raise RuntimeError("Streaming finished without any output")

                    usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=total_generated,
                        total_tokens=prompt_tokens + total_generated,
                        tokens_per_second=tokens_per_second,
                        processing_time=generation_time,
                        first_token_time=last_output.first_token_time,
                    )

                    final_chunk = ChatCompletionStreamResponse(
                        model=request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(content="", role="assistant"),
                                finish_reason="stop",
                            )
                        ],
                        usage=usage,
                    )

                    yield f"data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n"
                    yield "data: [DONE]\n\n"

                    self.metrics.total_tokens_generated += total_generated
                    self._record_api_key_usage(raw_request, prompt_tokens, total_generated)

                except Exception as e:
                    self._mark_stream_failure()
                    logger.exception(f"Error during streaming: {e}")
                    error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)
                    yield f"data: {error_response.body.decode()}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )

    async def health_check(self, raw_request: Request) -> JSONResponse:
        """Health check endpoint.

        Returns server health status and model information.

        Returns:
            JSONResponse with:
            - status: Current server status
            - timestamp: Current time
            - uptime_seconds: Server uptime
            - models: Loaded model information
            - active_requests: Current request count

            Status code 200 if READY, 503 otherwise.
        """
        self._authorize_request(raw_request)
        self.metrics.uptime_seconds = time.time() - self.metrics.start_time

        model_health_info = {}
        for name, adapter in self.adapters.items():
            model_health_info[name] = {
                "loaded": True,
                "type": adapter.get_model_info()["type"],
                "architecture": adapter.get_model_info()["architecture"],
                "max_model_len": adapter.get_model_info()["max_model_len"],
            }

        health_status = {
            "status": self.status.value,
            "timestamp": time.time(),
            "uptime_seconds": self.metrics.uptime_seconds,
            "models": model_health_info,
            "active_requests": len(self._active_requests),
        }

        status_code = 200 if self.status == ServerStatus.READY else 503
        return JSONResponse(health_status, status_code=status_code)

    async def get_metrics(self, raw_request: Request) -> JSONResponse:
        """Get server performance metrics.

        Returns:
            JSONResponse with comprehensive server metrics including
            request counts, token statistics, throughput, and status.
        """
        self._authorize_request(raw_request, endpoint="/v1/metrics")
        self.metrics.uptime_seconds = time.time() - self.metrics.start_time

        # Get authentication statistics
        auth_stats = self.auth_manager.get_statistics()

        return JSONResponse(
            {
                "uptime_seconds": self.metrics.uptime_seconds,
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "total_tokens_generated": self.metrics.total_tokens_generated,
                "average_tokens_per_second": round(self.metrics.average_tokens_per_second, 2),
                "active_requests": len(self._active_requests),
                "models_loaded": len(self.adapters),
                "status": self.status.value,
                "auth_stats": auth_stats,
            }
        )

    async def list_models(self, raw_request: Request) -> JSONResponse:
        """List available models.

        OpenAI-compatible model listing endpoint.

        Returns:
            JSONResponse with list of available models and their metadata.
        """
        self._authorize_request(raw_request)
        models_data = []
        for model_id, adapter in self.adapters.items():
            model_info = adapter.get_model_info()
            models_data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(self.metrics.start_time),
                    "owned_by": "easydel",
                    "max_model_len": adapter.esurge.max_model_len,
                    "metadata": {
                        **model_info,
                        "supports_chat": hasattr(adapter.processor, "apply_chat_template"),
                        "supports_function_calling": self.enable_function_calling,
                    },
                }
            )

        return JSONResponse(
            {
                "object": "list",
                "data": models_data,
                "total": len(models_data),
            }
        )

    async def get_model(self, model_id: str, raw_request: Request) -> JSONResponse:
        """Get model details.

        Args:
            model_id: Model identifier.

        Returns:
            JSONResponse with model metadata.

        Raises:
            HTTPException: If model not found.
        """
        self._authorize_request(raw_request)
        adapter = self._get_adapter(model_id)
        model_info = adapter.get_model_info()

        return JSONResponse(
            {
                "id": model_id,
                "object": "model",
                "created": int(self.metrics.start_time),
                "owned_by": "easydel",
                "metadata": {
                    **model_info,
                    "supports_chat": hasattr(adapter.processor, "apply_chat_template"),
                    "supports_function_calling": self.enable_function_calling,
                },
            }
        )

    async def list_tools(self, raw_request: Request) -> JSONResponse:
        """List available tools/functions for each model.

        Returns example tool definitions and supported formats.
        This is a placeholder that can be extended with actual tools.

        Returns:
            JSONResponse with tool definitions per model.
        """
        self._authorize_request(raw_request)
        model_names = list(self.adapters.keys())
        tools_response = self.create_tools_response(model_names)
        return JSONResponse(tools_response)

    async def _create_standard_response(
        self,
        request: ChatCompletionRequest,
        output: tp.Any,
        prompt_tokens: int,
        start_time: float,
    ) -> ChatCompletionResponse:
        """Create standard response without function calling."""
        completion = output.outputs[0]
        completion_tokens = len(completion.token_ids)
        generation_time = time.time() - start_time
        tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0

        message = ChatMessage(role="assistant", content=completion.text)

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tokens_per_second=tokens_per_second,
            processing_time=generation_time,
        )

        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionResponseChoice(index=0, message=message, finish_reason="stop")],
            usage=usage,
        )

    async def execute_tool(self, raw_request: Request) -> JSONResponse:
        """Execute a tool/function call.

        Placeholder endpoint for tool execution. Implement this method
        to integrate with actual tool execution systems.

        Args:
            raw_request: Tool execution request.

        Returns:
            JSONResponse with NOT_IMPLEMENTED status.

        Note:
            This is a placeholder that should be implemented based on
            specific tool execution requirements.
        """
        self._authorize_request(raw_request)
        return self.create_tool_execution_placeholder()

    @property
    def _endpoints(self) -> list:
        """Define all API endpoints including admin auth endpoints.

        Extends the base endpoints with admin authentication endpoints
        for API key management.

        Returns:
            List of EndpointConfig objects defining all server endpoints.
        """
        from ...inference_engine_interface import EndpointConfig

        # Get base endpoints from parent class
        base_endpoints = super()._endpoints

        # Add admin authentication endpoints
        admin_endpoints = [
            EndpointConfig(
                path="/v1/admin/keys",
                handler=self.create_api_key_endpoint,
                methods=["POST"],
                tags=["Admin", "Authentication"],
                summary=(
                    "Provision a fresh API key (optionally scoped via rate limits,"
                    "quotas, model filters, and metadata) and return the raw secret"
                    "exactly once. Use this endpoint during bootstrap or whenever new"
                    "tenants/on-call operators need access.\n\n"
                    "Because the raw key is only shown in the creation response, callers"
                    "should immediately store or display it securely—subsequent GET"
                    "requests only expose metadata."
                ),
            ),
            EndpointConfig(
                path="/v1/admin/keys",
                handler=self.list_api_keys_endpoint,
                methods=["GET"],
                tags=["Admin", "Authentication"],
                summary=(
                    "Enumerate every API key with pagination-friendly metadata (role,"
                    "status, usage counters) so administrators can audit who has access."
                    "The listing honors optional role/status filters, making it easy to"
                    "surface suspended or expiring credentials.\n\n"
                    "Use this endpoint to drive web consoles or to export inventories for"
                    "compliance reviews without digging into storage backends."
                ),
            ),
            EndpointConfig(
                path="/v1/admin/keys/{key_id}",
                handler=self.get_api_key_endpoint,
                methods=["GET"],
                tags=["Admin", "Authentication"],
                summary=(
                    "Fetch the full metadata record for a specific key ID—including role,"
                    "quotas, rate limits, and audit timestamps—without ever returning the"
                    "raw secret. Handy for incident response or when verifying what a"
                    "given credential is allowed to do.\n\n"
                    "Errors use standard 404 semantics, which makes it trivial to integrate"
                    "with scripts that verify key lifecycle."
                ),
            ),
            EndpointConfig(
                path="/v1/admin/keys/{key_id}",
                handler=self.update_api_key_endpoint,
                methods=["PATCH"],
                tags=["Admin", "Authentication"],
                summary=(
                    "Patch mutable properties of an API key (name, description, role,"
                    "rate-limit/quota policies, tags, metadata) in a single request."
                    "Only the provided fields change; everything else remains untouched.\n\n"
                    "Use this when rotating access levels or tightening quotas without"
                    "generating a brand new key."
                ),
            ),
            EndpointConfig(
                path="/v1/admin/keys/{key_id}/revoke",
                handler=self.revoke_api_key_endpoint,
                methods=["DELETE"],
                tags=["Admin", "Authentication"],
                summary=(
                    "Permanently revoke a key so it can no longer authenticate. The action"
                    "is logged with the admin's identity, making it suitable for emergency"
                    "credential revocation.\n\n"
                    "Unlike soft suspension, revocation is final—clients must request a new"
                    "key to regain access."
                ),
            ),
            EndpointConfig(
                path="/v1/admin/keys/{key_id}/suspend",
                handler=self.suspend_api_key_endpoint,
                methods=["POST"],
                tags=["Admin", "Authentication"],
                summary=(
                    "Temporarily disable a key without deleting its metadata. Suspended"
                    "keys can later be reactivated, which is useful for responding to"
                    "incidents or billing issues while preserving audit history.\n\n"
                    "The response confirms the action so operators can update ticketing"
                    "systems or dashboards."
                ),
            ),
            EndpointConfig(
                path="/v1/admin/keys/{key_id}/reactivate",
                handler=self.reactivate_api_key_endpoint,
                methods=["POST"],
                tags=["Admin", "Authentication"],
                summary=(
                    "Lift a prior suspension and restore the key to active status. Use this"
                    "after resolving the condition that triggered suspension (payment,"
                    "abuse investigation, etc.).\n\n"
                    "The endpoint keeps rate limits/quota settings intact, so reactivation"
                    "is a true toggle without additional configuration work."
                ),
            ),
            EndpointConfig(
                path="/v1/admin/keys/{key_id}",
                handler=self.delete_api_key_endpoint,
                methods=["DELETE"],
                tags=["Admin", "Authentication"],
                summary=(
                    "Remove a key and all associated metadata from storage. Choose this"
                    "route when cleaning up stale test credentials or when regulations"
                    "require full data deletion.\n\n"
                    "Because the action is irreversible, clients should double-check key"
                    "IDs before calling the endpoint."
                ),
            ),
            EndpointConfig(
                path="/v1/admin/keys/{key_id}/rotate",
                handler=self.rotate_api_key_endpoint,
                methods=["POST"],
                tags=["Admin", "Authentication"],
                summary=(
                    "Issue a brand new secret for an existing key record while preserving"
                    "its metadata, quotas, and audit trail. Rotations are ideal for routine"
                    "maintenance or when the secret may have leaked.\n\n"
                    "The response returns the new raw key exactly once—clients must store"
                    "it immediately before the value is discarded."
                ),
            ),
            EndpointConfig(
                path="/v1/admin/keys/stats",
                handler=self.get_api_key_stats_endpoint,
                methods=["GET"],
                tags=["Admin", "Authentication"],
                summary=(
                    "Provide aggregate counts and usage histograms across all keys so"
                    "operators can understand growth, saturation, or abuse patterns."
                    "Great for dashboards or alerting systems that watch for sudden spikes"
                    "in key creation or suspension events.\n\n"
                    "Because the endpoint is read-only, it can be safely wired into cron"
                    "jobs that feed BI tooling."
                ),
            ),
            EndpointConfig(
                path="/v1/admin/audit-logs",
                handler=self.get_audit_logs_endpoint,
                methods=["GET"],
                tags=["Admin", "Audit"],
                summary=(
                    "Return the chronological audit log covering key creation, updates,"
                    "revocations, and other sensitive events. Supports filters for key_id"
                    "and action type so investigations can focus on a narrow slice.\n\n"
                    "Teams should integrate this endpoint with SIEM pipelines or simply"
                    "download logs during compliance reviews."
                ),
            ),
        ]

        return base_endpoints + admin_endpoints
