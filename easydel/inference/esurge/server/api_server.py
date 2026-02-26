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

"""FastAPI server for eSurge with OpenAI API compatibility.

This module implements a production-ready API server that exposes eSurge inference
engines through OpenAI-compatible REST endpoints. It provides comprehensive features
for serving large language models in production environments.

Key Features:
    - Full OpenAI API v1 compatibility (/v1/chat/completions, /v1/completions)
    - Multi-model support with automatic routing based on model name
    - Streaming responses using Server-Sent Events (SSE)
    - Function/tool calling with pluggable parsers (Hermes, Qwen, etc.)
    - Production-grade authentication with RBAC, rate limiting, and audit logging
    - Real-time metrics and health monitoring
    - Thread-safe request handling with configurable concurrency limits

Architecture:
    The server uses an adapter pattern to bridge eSurge engines with the FastAPI
    infrastructure. The main components are:

    - eSurgeAdapter: Wraps eSurge instances to implement InferenceEngineAdapter
    - eSurgeApiServer: Main server class combining base server, tool calling,
      and auth endpoint mixins

Type Aliases:
    RefineSamplingParamsFn: Callback type for customizing sampling parameters.
    RefineChatRequestFn: Callback type for preprocessing chat requests.

Example:
    Basic single-model server::

        from easydel.inference.esurge import eSurge
        from easydel.inference.esurge.server import eSurgeApiServer

        esurge = eSurge.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        server = eSurgeApiServer(esurge, require_api_key=True)
        server.run(host="0.0.0.0", port=8000)

    Multi-model server with custom configurations::

        esurge_map = {
            "llama-7b": eSurge.from_pretrained("model-a"),
            "llama-13b": eSurge.from_pretrained("model-b"),
        }
        server = eSurgeApiServer(
            esurge_map,
            enable_function_calling=True,
            tool_parser_name="hermes",
            require_api_key=True,
            admin_key="sk-admin-secret",
        )
        server.run()

See Also:
    - `easydel.inference.esurge.esurge_engine.eSurge`: Core inference engine
    - `easydel.inference.inference_engine_interface`: Base server infrastructure
    - `easydel.workers.esurge.auth`: Authentication and authorization system
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
import typing as tp
import uuid
from collections.abc import AsyncGenerator, Iterable, Iterator
from http import HTTPStatus

from eformer.loggings import get_logger
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import ProcessorMixin

from easydel.workers.esurge.auth import (
    ApiKeyRole,
    EnhancedApiKeyManager,
    PermissionDenied,
    QuotaExceeded,
    RateLimitExceeded,
)

from ...inference_engine_interface import (
    BaseInferenceApiServer,
    InferenceEngineAdapter,
    ServerStatus,
    create_error_response,
)
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
    ResponsesRequest,
    UsageInfo,
)
from ...sampling_params import SamplingParams
from ...tools.tool_calling_mixin import ToolCallingMixin
from ..esurge_engine import RequestOutput, eSurge
from .auth_endpoints import AuthEndpointsMixin

logger = get_logger("eSurgeApiServer")

__all__ = (
    "RefineChatRequestFn",
    "RefineSamplingParamsFn",
    "ServerStatus",
    "create_error_response",
    "eSurgeAdapter",
    "eSurgeApiServer",
)

_STREAM_DATA = "data"
_STREAM_ERROR = "error"
_STREAM_END = "end"


RefineSamplingParamsFn = tp.Callable[
    [SamplingParams, ChatCompletionRequest | CompletionRequest, "eSurge"],
    SamplingParams | None,
]
"""Type alias for sampling parameter refinement callbacks.

A callable that receives the initial SamplingParams, the original request,
and the eSurge instance, and returns modified SamplingParams or None to
keep the original.
"""

RefineChatRequestFn = tp.Callable[[ChatCompletionRequest], ChatCompletionRequest | None]
"""Type alias for chat request preprocessing callbacks.

A callable that receives a ChatCompletionRequest and returns a modified
request or None to keep the original.
"""


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
    ) -> list[RequestOutput] | AsyncGenerator[RequestOutput, None]:
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
            loop = asyncio.get_running_loop()
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
            Dictionary containing model metadata: name, type, max_model_len,
            and max_num_seqs.
        """
        return {
            "name": self._model_name,
            "type": "esurge",
            "max_model_len": self.esurge.max_model_len,
            "max_num_seqs": self.esurge.max_num_seqs,
        }

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            The model name string assigned during adapter initialization.
        """
        return self._model_name

    @property
    def processor(self) -> tp.Any:
        """Return the tokenizer/processor associated with the eSurge instance.

        Returns:
            The tokenizer or processor object used by the underlying eSurge
            engine for text encoding and decoding.
        """
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
        response_store_worker_client: tp.Any | None = None,
        max_concurrent_generations: int | None = None,
        overload_message: str = "Server is busy, please try again later",
        extra_stops: str | list[str] | None = None,
        refine_sampling_params: RefineSamplingParamsFn | None = None,
        refine_chat_request: RefineChatRequestFn | None = None,
        enable_response_store: bool = True,
        default_store_responses: bool = True,
        max_stored_responses: int = 10_000,
        max_stored_conversations: int = 1_000,
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
            response_store_worker_client: Optional ResponseStoreWorkerClient instance for persistent /v1/responses state.
            tool_parser_worker_client: Optional ToolParserWorkerClient instance for ZMQ-based tool parsing. If None and use_tool_parser_worker=True, spawns a worker automatically.
            use_tool_parser_worker: If True and enable_function_calling=True, automatically spawns a tool parser ZMQ worker (default: True).
            max_concurrent_generations: Maximum concurrent inference jobs allowed. Defaults to the smallest
                ``max_num_seqs`` across loaded eSurge instances when not provided.
            overload_message: Custom error message returned when all generation slots are busy.
            extra_stops: Global stop strings applied to every request in addition to request-level stop values.
                Useful for enforcing server-side delimiters (for example ``\"<user>\"``) without requiring
                clients to set ``stop`` on each call.
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
            if getattr(esurge, "distributed_mode", False) and getattr(esurge, "distributed_role", None) == "worker":
                raise ValueError(
                    f"Model '{name}' is configured as distributed worker rank "
                    f"{getattr(esurge, 'distributed_rank', '?')}; only distributed leader rank "
                    "can run eSurgeApiServer."
                )
            self.adapters[name] = eSurgeAdapter(esurge, name)
            model_processors[name] = esurge.tokenizer

        # Initialize tool parsers using mixin
        self.tool_parsers = self.initialize_tool_parsers(
            model_processors=model_processors,
            tool_parser_name=tool_parser_name,
            enable_function_calling=enable_function_calling,
        )

        self.oai_like_processor = oai_like_processor
        self.tool_parser_name = tool_parser_name
        self.model_processors = model_processors
        self._refine_sampling_params_callback = refine_sampling_params
        self._refine_chat_request_callback = refine_chat_request
        self._extra_stops = self._normalize_stop_sequences(extra_stops)

        if max_concurrent_generations is not None:
            try:
                max_slots = int(max_concurrent_generations)
            except (TypeError, ValueError):
                max_slots = 0
        else:
            inferred = [esurge.max_num_seqs for esurge in esurge_map.values() if hasattr(esurge, "max_num_seqs")]
            max_slots = min(inferred) if inferred else 0
        max_slots = max(0, max_slots)

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
            max_concurrent_generations=max_slots,
            overload_message=overload_message,
            enable_response_store=enable_response_store,
            default_store_responses=default_store_responses,
            max_stored_responses=max_stored_responses,
            max_stored_conversations=max_stored_conversations,
            response_store_client=response_store_worker_client,
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
        """Extract API key from various request locations.

        Searches for an API key in the following locations (in order):
        1. Authorization header with "Bearer " prefix
        2. X-API-Key header
        3. "api_key" query parameter
        4. "user" query parameter (OpenAI compatibility)

        Args:
            raw_request: The incoming FastAPI request object.

        Returns:
            The extracted API key string if found, None otherwise.
        """
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
        """Extract API key candidates from the request JSON payload.

        Searches for API keys embedded within the request body, checking
        various field names commonly used for authentication. This allows
        clients to pass API keys in the request body as an alternative to
        headers or query parameters.

        Args:
            payload: The request payload, either as a Pydantic model or dict.
                For Pydantic models, checks `model_extra` for extra fields.

        Returns:
            List of unique, non-empty API key candidate strings found in
            the payload. The list preserves discovery order.

        Note:
            Checked field names include: api_key, apiKey, api-key, x-api-key,
            auth, authorization, token, key, and user (for OpenAI compatibility).
        """
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
        """Determine whether authentication enforcement is active.

        Checks multiple conditions to determine if the authentication system
        should enforce API key validation:
        1. Whether an auth_manager exists
        2. Whether the manager has an `enabled` property set
        3. Whether the manager supports `authorize_request` method
        4. Falls back to the `_require_api_key` setting

        Returns:
            True if authentication should be enforced, False otherwise.
        """

        try:
            manager = self.auth_manager
        except AttributeError:
            return False

        if manager is None:
            return False

        try:
            enabled = manager.enabled
        except AttributeError:
            enabled = None

        if enabled is not None:
            return bool(enabled)

        try:
            _ = manager.authorize_request
        except AttributeError:
            return self._require_api_key

        return True

    def _authorize_request(
        self,
        raw_request: Request,
        payload_api_keys: str | Iterable[str | None] | None = None,
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
            require_key = self.auth_manager.require_api_key
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
        """Record token usage statistics for the authenticated API key.

        Updates the auth manager's usage tracking for the API key that was
        used to authenticate the request. This enables per-key usage monitoring
        and quota enforcement.

        Args:
            raw_request: The FastAPI request object containing the authenticated
                API key in `request.state.api_key`. If None, no action is taken.
            prompt_tokens: Number of tokens in the input prompt.
            completion_tokens: Number of tokens generated in the response.

        Note:
            This method silently returns if no API key was used or if the
            request object is None. It should be called after successful
            completion of a request.
        """
        if raw_request is None:
            return

        api_key = getattr(raw_request.state, "api_key", None)
        if not api_key:
            return

        self.auth_manager.record_usage(api_key, prompt_tokens, completion_tokens)

    def _infer_sequence_length_from_engine(self, esurge: eSurge | None) -> int:
        """Infer the maximum sequence length from an eSurge engine.

        Attempts to determine the maximum sequence length for token generation
        by examining the engine's configuration. This is used as a default when
        `max_tokens` is not explicitly specified in requests.

        Args:
            esurge: The eSurge engine instance to query. If None, falls back to
                the first available adapter.

        Returns:
            The maximum model length if determinable, otherwise 128 as a
            conservative default.

        Note:
            The fallback value of 128 is intentionally conservative to prevent
            unexpected resource consumption when model configuration is unavailable.
        """
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
        """Ensure the request has a valid max_tokens value.

        Validates and normalizes the `max_tokens` field in the request,
        inferring a reasonable default from the engine configuration when
        the value is missing, invalid, or negative.

        This method also tracks whether the value was auto-inferred or
        explicitly provided by setting `request._auto_max_tokens`.

        Args:
            request: The chat or completion request to validate.
            esurge: Optional eSurge engine to query for default max length.

        Returns:
            The validated max_tokens value (either from request or inferred).

        Note:
            The method modifies the request object in-place when inferring
            a value, and sets `_auto_max_tokens` flag for downstream processing.
        """
        max_tokens_raw = getattr(request, "max_tokens", None)
        inferred_value: int | None = None

        def _mark_auto(value: int) -> int:
            try:
                request._auto_max_tokens = True
            except (AttributeError, ValueError):
                logger.debug("Unable to mark request as auto max_tokens.", exc_info=True)
            return value

        def _mark_manual() -> None:
            try:
                request._auto_max_tokens = False
            except (AttributeError, ValueError):
                logger.debug("Unable to mark request as manual max_tokens.", exc_info=True)

        def _infer_and_assign() -> int:
            nonlocal inferred_value
            if inferred_value is None:
                inferred_value = self._infer_sequence_length_from_engine(esurge)
            try:
                request.max_tokens = inferred_value
            except (AttributeError, ValueError):
                logger.debug("Unable to set inferred max_tokens on request; continuing with inferred value.")
            return _mark_auto(inferred_value)

        if max_tokens_raw is None:
            return _infer_and_assign()

        try:
            max_tokens_int = int(max_tokens_raw)
        except (TypeError, ValueError):
            logger.debug("Invalid max_tokens=%s supplied; inferring from engine.", max_tokens_raw)
            return _infer_and_assign()

        if max_tokens_int < 0:
            return _infer_and_assign()

        _mark_manual()
        return max_tokens_int

    def _prepare_sampling_params(
        self,
        request: ChatCompletionRequest | CompletionRequest,
        esurge: eSurge,
    ) -> SamplingParams:
        """Create and optionally refine sampling parameters for generation.

        Builds SamplingParams from the request and applies any user-provided
        refinement callback to customize generation behavior.

        Args:
            request: The chat or completion request containing generation settings.
            esurge: The eSurge engine instance (passed to refinement callback).

        Returns:
            SamplingParams configured for the generation, potentially modified
            by the refinement callback if one was provided during server init.
        """
        sampling_params = self._create_sampling_params(request)
        if self._refine_sampling_params_callback:
            refined = self._refine_sampling_params_callback(sampling_params, request, esurge)
            if refined is not None:
                sampling_params = refined
        sampling_params = self._apply_extra_stops_to_sampling_params(sampling_params)
        return sampling_params

    @staticmethod
    def _looks_like_tool_protocol_text(text: str | None) -> bool:
        """Return True when text appears to be tool protocol/control markup."""

        if not text:
            return False
        normalized = text.lower()
        markers = (
            "<tool_call",
            "<tool",
            "</tool_call>",
            "</tool",
            "<arg_key>",
            "<arg",
            "</arg_key>",
            "</arg",
            "<arg_value>",
            "</arg_value>",
            "<|observation|>",
            "<|observ",
            "observation|>",
            "<|assistant|>",
            "<|assist",
        )
        if any(marker in normalized for marker in markers):
            return True
        return ("<" in normalized) and ("tool" in normalized or "arg_" in normalized or "observation" in normalized)

    @staticmethod
    def _stream_debug_preview(text: tp.Any, *, max_chars: int = 160) -> str | None:
        """Return a compact escaped preview for debug logging."""

        if not isinstance(text, str):
            return None
        escaped = text.replace("\n", "\\n").replace("\r", "\\r")
        if len(escaped) <= max_chars:
            return escaped
        return f"{escaped[:max_chars]}..."

    @staticmethod
    def _stream_debug_len(value: tp.Any) -> int | None:
        """Return length for strings/sequences/mappings when available."""

        if value is None:
            return None
        if isinstance(value, (str, list, tuple, dict)):
            return len(value)
        return None

    @classmethod
    def _build_stream_debug_context(
        cls,
        *,
        endpoint: str,
        request_id: str | None,
        model: str | None,
        queue_kind: str | None = None,
        disconnected: bool | None = None,
        output: RequestOutput | None = None,
        last_output: RequestOutput | None = None,
        previous_text: tp.Any = None,
        current_text: tp.Any = None,
        delta_text: tp.Any = None,
        previous_token_ids: tp.Any = None,
        current_token_ids: tp.Any = None,
        delta_token_ids: tp.Any = None,
        raw_delta_message: tp.Any = None,
        delta_message: tp.Any = None,
        delta_tool_calls_raw: tp.Any = None,
        saw_tool_call_delta: bool | None = None,
        saw_function_call_delta: bool | None = None,
        stream_error: Exception | None = None,
        tools: tp.Any = None,
        messages: tp.Any = None,
    ) -> dict[str, tp.Any]:
        """Build a bounded debug payload for streaming failures."""

        observed_output = output or last_output
        primary_output = (
            observed_output.outputs[0] if (observed_output is not None and observed_output.outputs) else None
        )
        context: dict[str, tp.Any] = {
            "endpoint": endpoint,
            "request_id": request_id,
            "model": model,
            "queue_kind": queue_kind,
            "disconnected": disconnected,
            "tools_type": type(tools).__name__ if tools is not None else None,
            "tools_len": cls._stream_debug_len(tools),
            "first_tool_type": (type(tools[0]).__name__ if isinstance(tools, list) and len(tools) > 0 else None),
            "messages_type": type(messages).__name__ if messages is not None else None,
            "messages_len": cls._stream_debug_len(messages),
            "stream_error_type": type(stream_error).__name__ if stream_error is not None else None,
            "stream_error_message": str(stream_error) if stream_error is not None else None,
            "raw_delta_message_type": type(raw_delta_message).__name__ if raw_delta_message is not None else None,
            "delta_message_type": type(delta_message).__name__ if delta_message is not None else None,
            "delta_tool_calls_raw_type": type(delta_tool_calls_raw).__name__
            if delta_tool_calls_raw is not None
            else None,
            "delta_tool_calls_raw_len": cls._stream_debug_len(delta_tool_calls_raw),
            "delta_text_type": type(delta_text).__name__ if delta_text is not None else None,
            "delta_text_len": cls._stream_debug_len(delta_text),
            "delta_text_preview": cls._stream_debug_preview(delta_text),
            "previous_text_len": cls._stream_debug_len(previous_text),
            "current_text_len": cls._stream_debug_len(current_text),
            "previous_text_preview": cls._stream_debug_preview(previous_text),
            "current_text_preview": cls._stream_debug_preview(current_text),
            "previous_token_ids_len": cls._stream_debug_len(previous_token_ids),
            "current_token_ids_len": cls._stream_debug_len(current_token_ids),
            "delta_token_ids_len": cls._stream_debug_len(delta_token_ids),
            "saw_tool_call_delta": saw_tool_call_delta,
            "saw_function_call_delta": saw_function_call_delta,
        }

        if isinstance(delta_message, DeltaMessage):
            context["delta_message_content_type"] = (
                type(delta_message.content).__name__ if delta_message.content is not None else None
            )
            context["delta_message_content_len"] = cls._stream_debug_len(delta_message.content)
            context["delta_message_content_preview"] = cls._stream_debug_preview(delta_message.content)
            context["delta_message_tool_calls_type"] = (
                type(delta_message.tool_calls).__name__ if delta_message.tool_calls is not None else None
            )
            context["delta_message_tool_calls_len"] = cls._stream_debug_len(delta_message.tool_calls)
            context["delta_message_reasoning_len"] = cls._stream_debug_len(delta_message.reasoning_content)

        if observed_output is not None:
            context["output_request_id"] = observed_output.request_id
            context["output_finished"] = observed_output.finished
            context["output_num_generated_tokens"] = int(observed_output.num_generated_tokens or 0)
            context["output_accumulated_text_len"] = cls._stream_debug_len(observed_output.accumulated_text)
            context["output_delta_text_len"] = cls._stream_debug_len(observed_output.delta_text)
            context["output_reasoning_len"] = cls._stream_debug_len(observed_output.reasoning_content)
            context["output_delta_reasoning_len"] = cls._stream_debug_len(observed_output.delta_reasoning_content)
            context["output_tool_calls_type"] = (
                type(observed_output.tool_calls).__name__ if observed_output.tool_calls else None
            )
            context["output_delta_tool_calls_type"] = (
                type(observed_output.delta_tool_calls).__name__ if observed_output.delta_tool_calls else None
            )
            context["output_primary_token_ids_len"] = (
                cls._stream_debug_len(primary_output.token_ids) if primary_output is not None else None
            )

        producer_tb = getattr(stream_error, "__stream_producer_traceback__", None) if stream_error is not None else None
        if isinstance(producer_tb, str):
            if len(producer_tb) > 6000:
                producer_tb = f"{producer_tb[:6000]}..."
            context["stream_error_producer_traceback"] = producer_tb

        if isinstance(messages, list) and messages:
            first_message = messages[0]
            if isinstance(first_message, dict):
                context["first_message_role"] = first_message.get("role")
                context["first_message_content_type"] = (
                    type(first_message.get("content")).__name__ if "content" in first_message else None
                )

            for message in messages:
                if not isinstance(message, dict):
                    continue
                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list) or not tool_calls:
                    continue
                first_tool_call = tool_calls[0]
                if isinstance(first_tool_call, dict):
                    context["message_tool_calls_type"] = type(tool_calls).__name__
                    context["message_tool_calls_len"] = len(tool_calls)
                    function_payload = first_tool_call.get("function")
                    if isinstance(function_payload, dict):
                        arguments = function_payload.get("arguments")
                        context["message_tool_call_arguments_type"] = type(arguments).__name__
                        context["message_tool_call_arguments_len"] = cls._stream_debug_len(arguments)
                break

        return context

    @staticmethod
    def _normalize_stop_sequences(stop: tp.Any) -> list[str]:
        """Normalize stop input into a de-duplicated list of non-empty strings."""

        if stop is None:
            return []
        if isinstance(stop, str):
            candidates = [stop]
        elif isinstance(stop, (list, tuple, set)):
            candidates = list(stop)
        else:
            candidates = [stop]

        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate is None:
                continue
            value = candidate if isinstance(candidate, str) else str(candidate)
            if value == "" or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _apply_extra_stops_to_sampling_params(self, sampling_params: SamplingParams) -> SamplingParams:
        """Merge server-level stop strings into request sampling parameters."""

        if not self._extra_stops:
            return sampling_params

        merged = self._normalize_stop_sequences(getattr(sampling_params, "stop", None))
        seen = set(merged)
        for stop in self._extra_stops:
            if stop in seen:
                continue
            seen.add(stop)
            merged.append(stop)
        sampling_params.stop = merged
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
        raw_max_tokens = getattr(request, "max_tokens", None)
        max_tokens: int | None
        auto_max_tokens_requested = bool(getattr(request, "_auto_max_tokens", False))
        if raw_max_tokens is None or auto_max_tokens_requested:
            max_tokens = None
        else:
            try:
                max_tokens = int(raw_max_tokens)
            except (TypeError, ValueError):
                logger.debug("Unable to parse max_tokens=%s; defaulting to auto.", raw_max_tokens)
                max_tokens = None
            else:
                if max_tokens < 0:
                    max_tokens = None
        raw_temperature: float | None = request.temperature
        if raw_temperature is None:
            temperature_f = 1.0
        else:
            temperature_f = float(raw_temperature)
        temperature = max(0.0, min(temperature_f, 2.0))

        raw_top_k = getattr(request, "top_k", None)
        if raw_top_k is None:
            top_k = 0
        else:
            top_k = int(raw_top_k)
        if top_k < 0:
            top_k = 0

        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            presence_penalty=float(request.presence_penalty or 0.0),
            frequency_penalty=float(request.frequency_penalty or 0.0),
            repetition_penalty=float(getattr(request, "repetition_penalty", 1.0)),
            top_k=top_k,
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
                content_parts = msg["content"]
                has_non_text = False
                for part in content_parts:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type", "")
                    if part_type in ("image", "image_url", "input_image", "video", "video_url", "input_video"):
                        has_non_text = True
                        break
                    if any(k in part for k in ("image", "image_url", "video", "video_url")):
                        has_non_text = True
                        break
                if has_non_text:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Multimodal content detected in messages, but the request was routed to the text-only "
                            "chat-template path. Ensure your content parts use supported types (e.g. `image_url`) "
                            "and that the server is configured with a processor for VLMs."
                        ),
                    )

                text_parts: list[str] = []
                for part in content_parts:
                    if isinstance(part, str):
                        if part:
                            text_parts.append(part)
                        continue
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type in ("text", "input_text"):
                            text_parts.append(str(part.get("text", "")))
                msg["content"] = " ".join([p for p in text_parts if p])

        try:
            if request.chat_template_kwargs is None:
                request.chat_template_kwargs = {}
            add_generation_prompt = request.chat_template_kwargs.pop("add_generation_prompt", True)
            return str(
                processor.apply_chat_template(
                    tokenize=False,
                    conversation=conversation,
                    add_generation_prompt=add_generation_prompt,
                    tools=self.extract_tools(request=request),
                    **request.chat_template_kwargs,
                )
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
        loop = asyncio.get_running_loop()
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

            messages = self._prepare_messages_for_engine(request, esurge)
            is_multimodal = self._messages_have_multimodal_content(messages)
            if not is_multimodal:
                messages = self._prepare_text_messages_for_chat(messages)

            if request.stream:
                return await self._handle_chat_streaming(request, esurge, messages, request_id, raw_request)
            return await self._handle_chat_completion(request, esurge, messages, request_id, raw_request)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            logger.exception(f"Error in chat completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    @staticmethod
    def _messages_have_multimodal_content(messages: list[dict[str, tp.Any]]) -> bool:
        """Check if the message list contains multimodal content (images or videos).

        Examines message content parts to detect the presence of image or video
        data that requires special handling through the multimodal processing
        pipeline.

        Args:
            messages: List of message dictionaries with content fields.

        Returns:
            True if any message contains image or video content, False otherwise.

        Note:
            Supported multimodal type indicators include:
            - image, image_url, input_image (for images)
            - video, video_url, input_video (for videos)
        """
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type in ("image", "image_url", "input_image", "video", "video_url", "input_video"):
                    return True
                if any(k in item for k in ("image", "image_url", "video", "video_url")):
                    return True
        return False

    def _prepare_messages_for_engine(self, request: ChatCompletionRequest, esurge: eSurge) -> list[dict[str, tp.Any]]:
        """Convert Pydantic request messages into dictionaries for the eSurge engine.

        Extracts messages from the request and applies any necessary format
        conversions for compatibility with the underlying processor.

        Args:
            request: The chat completion request containing messages.
            esurge: The eSurge engine instance with the processor/tokenizer.

        Returns:
            List of message dictionaries ready for engine processing.
            Messages may be converted to OpenAI format if using a
            ProcessorMixin with `oai_like_processor` enabled.
        """
        conversation = request.model_dump(exclude_unset=True)["messages"]
        processor = esurge.tokenizer

        if isinstance(processor, ProcessorMixin) and self.oai_like_processor:
            from easydel.trainers.prompt_utils import convert_to_openai_format

            conversation = convert_to_openai_format(conversation)

        return tp.cast(list[dict[str, tp.Any]], conversation)

    @staticmethod
    def _prepare_text_messages_for_chat(messages: list[dict[str, tp.Any]]) -> list[dict[str, tp.Any]]:
        """Normalize text-only messages into plain string content for chat templates."""

        normalized: list[dict[str, tp.Any]] = []
        for msg in messages:
            msg_copy = dict(msg)
            content = msg_copy.get("content")
            if not isinstance(content, list):
                normalized.append(msg_copy)
                continue

            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    if part:
                        text_parts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    text = part.get("text", part.get("content", ""))
                    if text:
                        text_parts.append(str(text))
            msg_copy["content"] = " ".join(part for part in text_parts if part)
            normalized.append(msg_copy)
        return normalized

    async def _handle_chat_completion_multimodal(
        self,
        request: ChatCompletionRequest,
        esurge: eSurge,
        messages: list[dict[str, tp.Any]],
        request_id: str,
        raw_request: Request,
    ) -> ChatCompletionResponse:
        """Backward-compatible wrapper around the unified chat completion path."""
        return await self._handle_chat_completion(request, esurge, messages, request_id, raw_request)

    async def _handle_chat_streaming_multimodal(
        self,
        request: ChatCompletionRequest,
        esurge: eSurge,
        messages: list[dict[str, tp.Any]],
        request_id: str,
        raw_request: Request,
    ) -> StreamingResponse:
        """Backward-compatible wrapper around the unified chat streaming path."""
        return await self._handle_chat_streaming(request, esurge, messages, request_id, raw_request)

    @staticmethod
    def _prompt_token_count_from_output(output: RequestOutput) -> int:
        """Extract the prompt token count from a RequestOutput.

        Handles both flat and nested token ID structures that may be present
        in the output depending on the processing mode (single vs batched).

        Args:
            output: The RequestOutput containing prompt token information.

        Returns:
            Total count of prompt tokens. Returns 0 if prompt_token_ids is
            not available or not in a recognized format.
        """
        prompt_ids = output.prompt_token_ids
        if isinstance(prompt_ids, list) and prompt_ids:
            first = prompt_ids[0]
            if isinstance(first, list):
                return sum(len(seg) for seg in prompt_ids)
        if isinstance(prompt_ids, list):
            return len(prompt_ids)
        return 0

    async def responses(self, request: ResponsesRequest, raw_request: Request) -> tp.Any:
        """Handle OpenAI Responses API requests.

        Implements the /v1/responses endpoint for the OpenAI Responses API,
        supporting conversation continuations via `previous_response_id` or
        `conversation` parameters, and optional response persistence.

        Args:
            request: The Responses API request containing input, model selection,
                and optional continuation/storage parameters.
            raw_request: The raw FastAPI request for authentication context.

        Returns:
            For non-streaming: A JSON object conforming to the Responses API spec
                with output text, usage stats, and metadata.
            For streaming: A StreamingResponse with SSE events following the
                Responses API streaming protocol.

        Raises:
            HTTPException: For validation errors (400), auth failures (401/403/429),
                or model not found (404).

        Note:
            The Responses API supports stateful conversations when
            `enable_response_store=True` is configured on the server.
        """

        response_id = f"resp_{uuid.uuid4().hex}"
        payload_api_keys = self._extract_payload_api_keys(request)
        payload = request.model_dump(exclude_none=True, exclude_unset=True)

        try:
            store_flag = payload.get("store")
            store_response = self._default_store_responses if store_flag is None else bool(store_flag)

            previous_response_id = payload.get("previous_response_id")
            if not isinstance(previous_response_id, str):
                previous_response_id = None
            else:
                previous_response_id = previous_response_id.strip() or None

            conversation_id = self._normalize_conversation_id(payload.get("conversation"))
            if previous_response_id and conversation_id:
                raise HTTPException(status_code=400, detail="Cannot use both 'previous_response_id' and 'conversation'")

            model = payload.get("model")
            if not isinstance(model, str) or not model.strip():
                raise HTTPException(400, "Field 'model' is required")

            adapter = self._get_adapter(model)
            esurge = adapter.esurge

            requested_tokens, max_tokens = self._parse_responses_max_tokens(payload, esurge)
            self._authorize_request(
                raw_request,
                payload_api_keys=payload_api_keys,
                endpoint="/v1/responses",
                model=model,
                requested_tokens=requested_tokens,
            )

            input_messages = self._responses_payload_to_messages(payload, include_instructions=False)
            if not input_messages:
                raise HTTPException(400, "Field 'input' (or 'messages') cannot be empty")

            instructions = payload.get("instructions")
            if not isinstance(instructions, str) or not instructions.strip():
                instructions = None
            else:
                instructions = instructions.strip()

            history_messages: list[dict[str, tp.Any]] = []
            if previous_response_id:
                if not self._enable_response_store:
                    raise HTTPException(
                        status_code=400, detail="previous_response_id requires enable_response_store=True"
                    )
                prev = await self._response_store_get_response(previous_response_id)
                if prev is None:
                    raise HTTPException(status_code=400, detail=f"Unknown previous_response_id '{previous_response_id}'")
                history_messages = tp.cast(list[dict[str, tp.Any]], prev.get("conversation", []))
            elif conversation_id:
                if not self._enable_response_store:
                    raise HTTPException(status_code=400, detail="conversation requires enable_response_store=True")
                conversation_history = (await self._response_store_get_conversation(conversation_id)) or []
                history_messages = list(conversation_history)

            # `full_messages` is the persisted conversation state for this response (excludes `instructions`).
            full_messages = list(history_messages) + list(input_messages)

            # `engine_messages` is what we send to the model (may include ephemeral instructions).
            engine_messages = list(full_messages)
            if instructions:
                engine_messages.insert(0, {"role": "system", "content": instructions})

            raw_tools, tools_for_template = self._extract_responses_tools(payload)
            tool_request: ChatCompletionRequest | None = None
            if raw_tools:
                try:
                    tool_request = ChatCompletionRequest.model_validate(
                        {
                            "model": model,
                            "messages": self._flatten_messages_to_text(engine_messages),
                            "tools": raw_tools,
                            "tool_choice": payload.get("tool_choice"),
                            "temperature": payload.get("temperature", 1.0),
                            "top_p": payload.get("top_p", 1.0),
                            "max_tokens": max_tokens,
                            "stream": bool(payload.get("stream", False)),
                        }
                    )
                except Exception:
                    tool_request = None

            sampling_params = self._create_sampling_params_from_responses(payload, max_tokens)
            sampling_params = self._apply_extra_stops_to_sampling_params(sampling_params)
            stream = bool(payload.get("stream", False))
            reasoning_summary_requested = self._responses_reasoning_summary_requested(payload)

            if not stream:
                async with self._acquire_generation_slot():
                    loop = asyncio.get_running_loop()

                    def _run_chat() -> RequestOutput:
                        return tp.cast(
                            RequestOutput,
                            esurge.chat(
                                messages=engine_messages,
                                tools=tools_for_template,
                                sampling_params=sampling_params,
                                request_id=response_id,
                                stream=False,
                            ),
                        )

                    try:
                        output = await loop.run_in_executor(self.thread_pool, _run_chat)
                    except ValueError as e:
                        raise HTTPException(status_code=400, detail=str(e)) from e

                completion_tokens = int(output.num_generated_tokens or 0)
                prompt_tokens = self._prompt_token_count_from_output(output)

                primary_output = output.outputs[0] if output.outputs else None
                response_text = output.accumulated_text or output.get_text()
                reasoning_text = output.reasoning_content or (
                    primary_output.reasoning_content if primary_output is not None else None
                )
                tool_calls_payload = self._jsonify_tool_calls(
                    output.tool_calls or (primary_output.tool_calls if primary_output is not None else None)
                )

                if tool_calls_payload is None and tool_request is not None:
                    message, _finish_reason = self.extract_tool_calls_batch(
                        response_text=response_text,
                        request=tool_request,
                        model_name=model,
                    )
                    tool_calls_payload = self._jsonify_tool_calls(message.model_extra.get("tool_calls"))
                    if tool_calls_payload is not None:
                        response_text = tp.cast(str, message.content) if message.content is not None else ""
                    elif message.content is not None:
                        response_text = tp.cast(str, message.content)
                if tool_calls_payload:
                    # Prefer canonical function_call items over raw protocol text.
                    response_text = ""

                output_items = self._build_responses_output_items(
                    output_text=response_text,
                    tool_calls=tool_calls_payload,
                    reasoning_text=reasoning_text,
                    include_reasoning_summary=reasoning_summary_requested,
                )

                self.metrics.total_tokens_generated += completion_tokens
                self._record_api_key_usage(raw_request, prompt_tokens, completion_tokens)

                response_obj = self._build_responses_object(
                    response_id=response_id,
                    model=model,
                    output_text=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    tool_calls=tool_calls_payload,
                    reasoning_text=reasoning_text,
                    include_reasoning_summary=reasoning_summary_requested,
                    output_items=output_items,
                )
                response_obj.update(
                    {
                        "error": None,
                        "incomplete_details": None,
                        "instructions": instructions,
                        "max_output_tokens": payload.get("max_output_tokens"),
                        "previous_response_id": previous_response_id,
                        "store": store_response,
                        "temperature": payload.get("temperature", 1.0),
                        "top_p": payload.get("top_p", 1.0),
                        "truncation": payload.get("truncation", "disabled"),
                        "tool_choice": payload.get("tool_choice", "auto"),
                        "tools": raw_tools or [],
                        "parallel_tool_calls": payload.get("parallel_tool_calls", True),
                        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                        "text": {"format": {"type": "text"}},
                    }
                )

                if store_response and self._enable_response_store:
                    assistant_turn = self._responses_assistant_message_from_output_items(output_items)
                    conversation_after = self._conversation_from_messages(full_messages, assistant_turn)
                    await self._response_store_put_response(
                        response_id,
                        {
                            "id": response_id,
                            "model": model,
                            "created_at": response_obj.get("created_at"),
                            "previous_response_id": previous_response_id,
                            "conversation_id": conversation_id,
                            "conversation": conversation_after,
                            "response": response_obj,
                        },
                    )
                    if conversation_id:
                        await self._response_store_put_conversation(conversation_id, conversation_after)

                return response_obj

            async def generate_stream():
                async with self._acquire_generation_slot():
                    previous_text = ""
                    previous_token_ids: list[int] = []
                    prompt_tokens = 0
                    completion_tokens = 0
                    last_output: RequestOutput | None = None
                    disconnected = False

                    output_items_stream: list[dict[str, tp.Any]] = []
                    next_output_index = 0

                    reasoning_item_id: str | None = None
                    reasoning_output_index: int | None = None
                    reasoning_text_accum = ""
                    reasoning_done = False

                    function_states: dict[str, dict[str, tp.Any]] = {}
                    function_order: list[str] = []
                    saw_function_call_delta = False

                    message_item: dict[str, tp.Any] | None = None
                    message_item_id: str | None = None
                    message_output_index: int | None = None
                    message_text_accum = ""
                    message_done = False
                    content_index = 0

                    yield self._sse_event(
                        "response.created",
                        {
                            "type": "response.created",
                            "response": {
                                "id": response_id,
                                "object": "response",
                                "created_at": int(time.time()),
                                "model": model,
                                "status": "in_progress",
                                "output": [],
                            },
                        },
                    )

                    queue = self._start_stream_task(
                        lambda: tp.cast(
                            Iterator[RequestOutput],
                            esurge.chat(
                                messages=engine_messages,
                                tools=tools_for_template,
                                sampling_params=sampling_params,
                                request_id=response_id,
                                stream=True,
                            ),
                        )
                    )

                    try:
                        stream_error: Exception | None = None  # pyright: ignore[reportUnusedVariable]
                        while True:
                            if await raw_request.is_disconnected():
                                try:
                                    esurge.abort_request(response_id)
                                except Exception:
                                    logger.debug(
                                        "Failed to abort response %s after disconnect", response_id, exc_info=True
                                    )
                                disconnected = True
                                break
                            kind, stream_payload = await queue.get()
                            if kind == _STREAM_END:
                                break
                            if kind == _STREAM_ERROR:
                                stream_error = tp.cast(Exception, stream_payload)  # pyright: ignore[reportUnusedVariable]
                                raise tp.cast(Exception, stream_payload)

                            output = tp.cast(RequestOutput, stream_payload)
                            last_output = output
                            if not prompt_tokens:
                                prompt_tokens = self._prompt_token_count_from_output(output)
                            completion_tokens = int(output.num_generated_tokens or 0)

                            primary_output = output.outputs[0] if output.outputs else None
                            current_text = output.accumulated_text or ""
                            delta_text = self._compute_delta_text(current_text, previous_text, output.delta_text or "")
                            delta_reasoning = output.delta_reasoning_content or ""
                            delta_tool_calls_raw = output.delta_tool_calls

                            engine_has_parsers = (
                                hasattr(esurge, "_tool_parser_class") and esurge._tool_parser_class is not None
                            ) or (
                                hasattr(esurge, "_reasoning_parser_class") and esurge._reasoning_parser_class is not None
                            )

                            current_token_ids = primary_output.token_ids if primary_output is not None else []
                            if engine_has_parsers:
                                previous_text = current_text
                                previous_token_ids = current_token_ids
                            elif tool_request is not None:
                                delta_token_ids = (
                                    current_token_ids[len(previous_token_ids) :]
                                    if previous_token_ids
                                    else current_token_ids
                                )
                                raw_delta_message = self.extract_tool_calls_streaming(
                                    model_name=model,
                                    previous_text=previous_text,
                                    current_text=current_text,
                                    delta_text=delta_text,
                                    previous_token_ids=previous_token_ids,
                                    current_token_ids=current_token_ids,
                                    delta_token_ids=delta_token_ids,
                                    request=tool_request,
                                )
                                delta_message = self._coerce_stream_delta_message(
                                    raw_delta_message,
                                    fallback_text=delta_text,
                                    default_role="assistant",
                                )
                                previous_text = current_text
                                previous_token_ids = current_token_ids

                                if delta_message is not None:
                                    if isinstance(delta_message.content, str):
                                        delta_text = delta_message.content
                                    elif delta_message.tool_calls:
                                        delta_text = ""
                                    delta_tool_calls_raw = delta_message.tool_calls
                            else:
                                previous_text = current_text
                                previous_token_ids = current_token_ids

                            current_reasoning = output.reasoning_content or (
                                primary_output.reasoning_content if primary_output is not None else ""
                            )
                            if (
                                reasoning_summary_requested
                                and isinstance(current_reasoning, str)
                                and len(current_reasoning) > len(reasoning_text_accum)
                                and not delta_reasoning
                            ):
                                delta_reasoning = current_reasoning[len(reasoning_text_accum) :]

                            if reasoning_summary_requested and delta_reasoning:
                                if reasoning_item_id is None:
                                    reasoning_item = self._build_responses_reasoning_item("")
                                    reasoning_item["status"] = "in_progress"
                                    reasoning_item_id = tp.cast(str, reasoning_item["id"])
                                    reasoning_output_index = next_output_index
                                    next_output_index += 1
                                    output_items_stream.append(reasoning_item)
                                    yield self._sse_event(
                                        "response.output_item.added",
                                        {
                                            "type": "response.output_item.added",
                                            "output_index": reasoning_output_index,
                                            "item": reasoning_item,
                                        },
                                    )

                                reasoning_text_accum += delta_reasoning
                                output_items_stream[reasoning_output_index]["summary"][0]["text"] = reasoning_text_accum
                                yield self._sse_event(
                                    "response.reasoning_summary_text.delta",
                                    {
                                        "type": "response.reasoning_summary_text.delta",
                                        "output_index": reasoning_output_index,
                                        "item_id": reasoning_item_id,
                                        "summary_index": 0,
                                        "delta": delta_reasoning,
                                    },
                                )

                            delta_tool_calls = self._jsonify_tool_calls(delta_tool_calls_raw) or []
                            if delta_tool_calls:
                                saw_function_call_delta = True
                            for position, delta_call in enumerate(delta_tool_calls):
                                if not isinstance(delta_call, dict):
                                    continue

                                call_index_raw = delta_call.get("index")
                                call_index = call_index_raw if isinstance(call_index_raw, int) else position

                                delta_call_id = delta_call.get("id")
                                if isinstance(delta_call_id, str) and delta_call_id:
                                    call_key = delta_call_id
                                    resolved_call_id = delta_call_id
                                else:
                                    call_key = f"idx:{call_index}"
                                    resolved_call_id = f"call_{uuid.uuid4().hex}"

                                state = function_states.get(call_key)
                                if state is None:
                                    function_item = {
                                        "id": f"fc_{uuid.uuid4().hex}",
                                        "type": "function_call",
                                        "call_id": resolved_call_id,
                                        "name": "",
                                        "arguments": "",
                                        "status": "in_progress",
                                    }
                                    state = {
                                        "item": function_item,
                                        "item_id": function_item["id"],
                                        "output_index": next_output_index,
                                        "done": False,
                                    }
                                    function_states[call_key] = state
                                    function_order.append(call_key)
                                    output_items_stream.append(function_item)
                                    next_output_index += 1
                                    yield self._sse_event(
                                        "response.output_item.added",
                                        {
                                            "type": "response.output_item.added",
                                            "output_index": state["output_index"],
                                            "item": function_item,
                                        },
                                    )

                                function_payload = delta_call.get("function")
                                if not isinstance(function_payload, dict):
                                    function_payload = {}

                                name = function_payload.get("name")
                                if isinstance(name, str) and name:
                                    state["item"]["name"] = name

                                arguments_delta = function_payload.get("arguments")
                                if arguments_delta is None:
                                    continue

                                if isinstance(arguments_delta, str):
                                    arguments_delta_text = arguments_delta
                                elif isinstance(arguments_delta, (dict, list)):
                                    arguments_delta_text = json.dumps(
                                        arguments_delta, ensure_ascii=False, separators=(",", ":")
                                    )
                                else:
                                    arguments_delta_text = str(arguments_delta)

                                state["item"]["arguments"] += arguments_delta_text
                                yield self._sse_event(
                                    "response.function_call_arguments.delta",
                                    {
                                        "type": "response.function_call_arguments.delta",
                                        "output_index": state["output_index"],
                                        "item_id": state["item_id"],
                                        "delta": arguments_delta_text,
                                    },
                                )

                            if raw_tools and self._looks_like_tool_protocol_text(delta_text):
                                delta_text = ""
                            if saw_function_call_delta:
                                delta_text = ""

                            if delta_text:
                                if message_item is None:
                                    message_item = {
                                        "id": f"msg_{uuid.uuid4().hex}",
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [],
                                        "status": "in_progress",
                                    }
                                    message_item_id = tp.cast(str, message_item["id"])
                                    message_output_index = next_output_index
                                    next_output_index += 1
                                    output_items_stream.append(message_item)
                                    yield self._sse_event(
                                        "response.output_item.added",
                                        {
                                            "type": "response.output_item.added",
                                            "output_index": message_output_index,
                                            "item": message_item,
                                        },
                                    )
                                    yield self._sse_event(
                                        "response.content_part.added",
                                        {
                                            "type": "response.content_part.added",
                                            "output_index": message_output_index,
                                            "item_id": message_item_id,
                                            "content_index": content_index,
                                            "part": {
                                                "type": "output_text",
                                                "annotations": [],
                                                "logprobs": [],
                                                "text": "",
                                            },
                                        },
                                    )

                                message_text_accum += delta_text
                                yield self._sse_event(
                                    "response.output_text.delta",
                                    {
                                        "type": "response.output_text.delta",
                                        "output_index": message_output_index,
                                        "item_id": message_item_id,
                                        "content_index": content_index,
                                        "delta": delta_text,
                                    },
                                )

                        if disconnected:
                            return

                        full_text = last_output.accumulated_text if last_output is not None else previous_text
                        primary_output = (
                            last_output.outputs[0] if (last_output is not None and last_output.outputs) else None
                        )
                        reasoning_text_final = last_output.reasoning_content if last_output is not None else None
                        if not reasoning_text_final and primary_output is not None:
                            reasoning_text_final = primary_output.reasoning_content

                        tool_calls_payload = self._jsonify_tool_calls(
                            (last_output.tool_calls if last_output is not None else None)
                            or (primary_output.tool_calls if primary_output is not None else None)
                        )

                        if tool_calls_payload is None and tool_request is not None and full_text:
                            message, _finish_reason = self.extract_tool_calls_batch(
                                response_text=full_text,
                                request=tool_request,
                                model_name=model,
                            )
                            tool_calls_payload = self._jsonify_tool_calls(message.model_extra.get("tool_calls"))
                            if tool_calls_payload is not None:
                                full_text = tp.cast(str, message.content) if message.content is not None else ""
                            elif message.content is not None:
                                full_text = tp.cast(str, message.content)
                        if tool_calls_payload or saw_function_call_delta:
                            full_text = ""

                        if (
                            reasoning_summary_requested
                            and not reasoning_text_accum
                            and isinstance(reasoning_text_final, str)
                            and reasoning_text_final.strip()
                        ):
                            reasoning_text_accum = reasoning_text_final

                        if last_output is not None:
                            completion_tokens = int(last_output.num_generated_tokens or completion_tokens)

                        if reasoning_summary_requested and reasoning_text_accum and not reasoning_done:
                            if reasoning_item_id is None:
                                reasoning_item = self._build_responses_reasoning_item(reasoning_text_accum)
                                reasoning_item["status"] = "in_progress"
                                reasoning_item_id = tp.cast(str, reasoning_item["id"])
                                reasoning_output_index = next_output_index
                                next_output_index += 1
                                output_items_stream.append(reasoning_item)
                                yield self._sse_event(
                                    "response.output_item.added",
                                    {
                                        "type": "response.output_item.added",
                                        "output_index": reasoning_output_index,
                                        "item": reasoning_item,
                                    },
                                )

                            output_items_stream[reasoning_output_index]["summary"][0]["text"] = reasoning_text_accum
                            output_items_stream[reasoning_output_index]["status"] = "completed"
                            yield self._sse_event(
                                "response.reasoning_summary_text.done",
                                {
                                    "type": "response.reasoning_summary_text.done",
                                    "output_index": reasoning_output_index,
                                    "item_id": reasoning_item_id,
                                    "summary_index": 0,
                                    "text": reasoning_text_accum,
                                },
                            )
                            yield self._sse_event(
                                "response.output_item.done",
                                {
                                    "type": "response.output_item.done",
                                    "output_index": reasoning_output_index,
                                    "item": output_items_stream[reasoning_output_index],
                                },
                            )
                            reasoning_done = True

                        normalized_tool_calls = tool_calls_payload or []
                        for idx, tool_call in enumerate(normalized_tool_calls):
                            if not isinstance(tool_call, dict):
                                continue
                            function_payload = tool_call.get("function")
                            if not isinstance(function_payload, dict):
                                continue
                            tool_call_id = tool_call.get("id")
                            if isinstance(tool_call_id, str) and tool_call_id:
                                call_key = tool_call_id
                                resolved_call_id = tool_call_id
                            else:
                                call_key = f"idx:{idx}"
                                resolved_call_id = f"call_{uuid.uuid4().hex}"

                            state = function_states.get(call_key)
                            if state is None:
                                function_item = {
                                    "id": f"fc_{uuid.uuid4().hex}",
                                    "type": "function_call",
                                    "call_id": resolved_call_id,
                                    "name": "",
                                    "arguments": "",
                                    "status": "in_progress",
                                }
                                state = {
                                    "item": function_item,
                                    "item_id": function_item["id"],
                                    "output_index": next_output_index,
                                    "done": False,
                                }
                                function_states[call_key] = state
                                function_order.append(call_key)
                                output_items_stream.append(function_item)
                                next_output_index += 1
                                yield self._sse_event(
                                    "response.output_item.added",
                                    {
                                        "type": "response.output_item.added",
                                        "output_index": state["output_index"],
                                        "item": function_item,
                                    },
                                )

                            name = function_payload.get("name")
                            if isinstance(name, str) and name:
                                state["item"]["name"] = name

                            arguments = function_payload.get("arguments", "")
                            if isinstance(arguments, str):
                                arguments_text = arguments
                            elif isinstance(arguments, (dict, list)):
                                arguments_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                            else:
                                arguments_text = str(arguments)

                            if arguments_text and not state["item"]["arguments"]:
                                state["item"]["arguments"] = arguments_text
                                yield self._sse_event(
                                    "response.function_call_arguments.delta",
                                    {
                                        "type": "response.function_call_arguments.delta",
                                        "output_index": state["output_index"],
                                        "item_id": state["item_id"],
                                        "delta": arguments_text,
                                    },
                                )
                            elif arguments_text:
                                state["item"]["arguments"] = arguments_text

                        for call_key in function_order:
                            state = function_states.get(call_key)
                            if state is None or state.get("done"):
                                continue
                            state["item"]["status"] = "completed"
                            yield self._sse_event(
                                "response.function_call_arguments.done",
                                {
                                    "type": "response.function_call_arguments.done",
                                    "output_index": state["output_index"],
                                    "item_id": state["item_id"],
                                    "arguments": state["item"]["arguments"],
                                },
                            )
                            yield self._sse_event(
                                "response.output_item.done",
                                {
                                    "type": "response.output_item.done",
                                    "output_index": state["output_index"],
                                    "item": state["item"],
                                },
                            )
                            state["done"] = True

                        if message_item is None:
                            message_item = {
                                "id": f"msg_{uuid.uuid4().hex}",
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "status": "in_progress",
                            }
                            message_item_id = tp.cast(str, message_item["id"])
                            message_output_index = next_output_index
                            next_output_index += 1
                            output_items_stream.append(message_item)
                            yield self._sse_event(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": message_output_index,
                                    "item": message_item,
                                },
                            )
                            yield self._sse_event(
                                "response.content_part.added",
                                {
                                    "type": "response.content_part.added",
                                    "output_index": message_output_index,
                                    "item_id": message_item_id,
                                    "content_index": content_index,
                                    "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": ""},
                                },
                            )

                        message_text_accum = full_text
                        yield self._sse_event(
                            "response.output_text.done",
                            {
                                "type": "response.output_text.done",
                                "output_index": message_output_index,
                                "item_id": message_item_id,
                                "content_index": content_index,
                                "text": full_text,
                            },
                        )
                        message_item["content"] = [
                            {"type": "output_text", "annotations": [], "logprobs": [], "text": full_text}
                        ]
                        message_item["status"] = "completed"
                        if not message_done:
                            yield self._sse_event(
                                "response.output_item.done",
                                {
                                    "type": "response.output_item.done",
                                    "output_index": message_output_index,
                                    "item": message_item,
                                },
                            )
                            message_done = True

                        final_obj = self._build_responses_object(
                            response_id=response_id,
                            model=model,
                            output_text=full_text,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            tool_calls=tool_calls_payload,
                            reasoning_text=reasoning_text_accum or None,
                            include_reasoning_summary=reasoning_summary_requested,
                            output_items=output_items_stream,
                        )
                        final_obj.update(
                            {
                                "error": None,
                                "incomplete_details": None,
                                "instructions": instructions,
                                "max_output_tokens": payload.get("max_output_tokens"),
                                "previous_response_id": previous_response_id,
                                "store": store_response,
                                "temperature": payload.get("temperature", 1.0),
                                "top_p": payload.get("top_p", 1.0),
                                "truncation": payload.get("truncation", "disabled"),
                                "tool_choice": payload.get("tool_choice", "auto"),
                                "tools": raw_tools or [],
                                "parallel_tool_calls": payload.get("parallel_tool_calls", True),
                                "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                                "text": {"format": {"type": "text"}},
                            }
                        )

                        if store_response and self._enable_response_store:
                            assistant_turn = self._responses_assistant_message_from_output_items(output_items_stream)
                            conversation_after = self._conversation_from_messages(full_messages, assistant_turn)
                            await self._response_store_put_response(
                                response_id,
                                {
                                    "id": response_id,
                                    "model": model,
                                    "created_at": final_obj.get("created_at"),
                                    "previous_response_id": previous_response_id,
                                    "conversation_id": conversation_id,
                                    "conversation": conversation_after,
                                    "response": final_obj,
                                },
                            )
                            if conversation_id:
                                await self._response_store_put_conversation(conversation_id, conversation_after)

                        yield self._sse_event(
                            "response.completed", {"type": "response.completed", "response": final_obj}
                        )

                        self.metrics.total_tokens_generated += completion_tokens
                        self._record_api_key_usage(raw_request, prompt_tokens, completion_tokens)

                    except Exception as e:
                        self._mark_stream_failure()
                        debug_context = self._build_stream_debug_context(
                            endpoint="/v1/responses",
                            request_id=response_id,
                            model=model,
                            queue_kind=locals().get("kind"),
                            disconnected=locals().get("disconnected"),
                            output=locals().get("output"),
                            last_output=locals().get("last_output"),
                            previous_text=locals().get("previous_text"),
                            current_text=locals().get("current_text"),
                            delta_text=locals().get("delta_text"),
                            previous_token_ids=locals().get("previous_token_ids"),
                            current_token_ids=locals().get("current_token_ids"),
                            delta_token_ids=locals().get("delta_token_ids"),
                            raw_delta_message=locals().get("raw_delta_message"),
                            delta_message=locals().get("delta_message"),
                            delta_tool_calls_raw=locals().get("delta_tool_calls_raw"),
                            saw_function_call_delta=locals().get("saw_function_call_delta"),
                            stream_error=locals().get("stream_error"),
                            tools=tools_for_template,
                            messages=locals().get("engine_messages"),
                        )
                        logger.exception("Error during /v1/responses streaming: %s | context=%s", e, debug_context)
                        error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), response_id)
                        yield self._sse_event(
                            "response.error",
                            {"type": "response.error", "error": json.loads(error_response.body.decode())},
                        )

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": response_id,
                },
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error in /v1/responses: %s", e)
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), response_id)

    def _build_chat_completion_response(
        self,
        request: ChatCompletionRequest,
        esurge: eSurge,
        output: RequestOutput,
        raw_request: Request,
    ) -> ChatCompletionResponse:
        """Build a ChatCompletionResponse from a finalized RequestOutput snapshot."""

        completion_tokens = int(output.num_generated_tokens or 0)
        prompt_tokens = self._prompt_token_count_from_output(output)
        self.metrics.total_tokens_generated += completion_tokens

        tokens_per_second = output.tokens_per_second
        processing_time = output.processing_time

        if self.metrics.average_tokens_per_second == 0:
            self.metrics.average_tokens_per_second = tokens_per_second
        else:
            self.metrics.average_tokens_per_second = (
                self.metrics.average_tokens_per_second * 0.9 + tokens_per_second * 0.1
            )

        response_text = output.accumulated_text or output.get_text()
        engine_has_parsers = (hasattr(esurge, "_tool_parser_class") and esurge._tool_parser_class is not None) or (
            hasattr(esurge, "_reasoning_parser_class") and esurge._reasoning_parser_class is not None
        )

        choices: list[ChatCompletionResponseChoice] = []
        for idx, completion in enumerate(output.outputs):
            if completion.tool_calls:
                message = ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=completion.tool_calls,
                    reasoning_content=completion.reasoning_content,
                )
                finish_reason = "tool_calls"
            elif completion.reasoning_content is not None or engine_has_parsers:
                message = ChatMessage(
                    role="assistant",
                    content=completion.text,
                    reasoning_content=completion.reasoning_content,
                )
                finish_reason = completion.finish_reason or "stop"
            else:
                message, finish_reason_extracted = self.extract_tool_calls_batch(
                    response_text=response_text,
                    request=request,
                    model_name=request.model,
                )
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

    async def _handle_chat_completion(
        self,
        request: ChatCompletionRequest,
        esurge: eSurge,
        messages: list[dict[str, tp.Any]],
        request_id: str,
        raw_request: Request,
    ) -> ChatCompletionResponse:
        """Handle non-streaming chat completion via eSurge.chat()."""

        async with self._acquire_generation_slot():
            sampling_params = self._prepare_sampling_params(request, esurge)
            loop = asyncio.get_running_loop()

            def _run_chat() -> RequestOutput:
                return tp.cast(
                    RequestOutput,
                    esurge.chat(
                        messages=messages,
                        tools=self.extract_tools(request=request),
                        sampling_params=sampling_params,
                        request_id=request_id,
                        stream=False,
                        chat_template_kwargs=request.chat_template_kwargs,
                    ),
                )

            try:
                output = await loop.run_in_executor(self.thread_pool, _run_chat)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

            return self._build_chat_completion_response(request, esurge, output, raw_request)

    async def _handle_chat_streaming(
        self,
        request: ChatCompletionRequest,
        esurge: eSurge,
        messages: list[dict[str, tp.Any]],
        request_id: str,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming chat completion via eSurge.chat()."""

        sampling_params = self._prepare_sampling_params(request, esurge)
        tools = self.extract_tools(request=request)

        async def generate_stream():
            async with self._acquire_generation_slot():
                prompt_tokens = 0
                tool_parser = self.get_tool_parser_for_model(request.model)
                previous_text = ""
                previous_token_ids: list[int] = []
                queue = self._start_stream_task(
                    lambda: tp.cast(
                        Iterator[RequestOutput],
                        esurge.chat(
                            messages=messages,
                            tools=tools,
                            sampling_params=sampling_params,
                            request_id=request_id,
                            stream=True,
                            chat_template_kwargs=request.chat_template_kwargs,
                        ),
                    )
                )
                total_generated = 0
                generation_time = 0.0
                tokens_per_second = 0.0
                last_output: RequestOutput | None = None
                disconnected = False
                saw_tool_call_delta = False

                try:
                    stream_error: Exception | None = None  # pyright: ignore[reportUnusedVariable]
                    while True:
                        if await raw_request.is_disconnected():
                            try:
                                esurge.abort_request(request_id)
                            except Exception:
                                logger.debug("Failed to abort request %s after disconnect", request_id, exc_info=True)
                            disconnected = True
                            break
                        kind, payload = await queue.get()
                        if kind == _STREAM_END:
                            break
                        if kind == _STREAM_ERROR:
                            stream_error = tp.cast(Exception, payload)  # pyright: ignore[reportUnusedVariable]
                            raise tp.cast(Exception, payload)

                        output = tp.cast(RequestOutput, payload)
                        last_output = output
                        if not prompt_tokens:
                            prompt_tokens = self._prompt_token_count_from_output(output)

                        current_completion_tokens = int(output.num_generated_tokens or 0)
                        current_tps = output.tokens_per_second
                        elapsed_time = output.processing_time

                        current_text = output.accumulated_text or ""
                        delta_text = self._compute_delta_text(current_text, previous_text, output.delta_text or "")

                        engine_has_parsers = (
                            hasattr(esurge, "_tool_parser_class") and esurge._tool_parser_class is not None
                        ) or (hasattr(esurge, "_reasoning_parser_class") and esurge._reasoning_parser_class is not None)

                        if engine_has_parsers:
                            previous_text = current_text
                            current_token_ids = output.outputs[0].token_ids if output.outputs else []
                            previous_token_ids = current_token_ids

                            has_parsed_content = (
                                output.delta_tool_calls or output.delta_reasoning_content or output.delta_text
                            )
                            if not has_parsed_content:
                                continue

                            delta_message = DeltaMessage(
                                role="assistant",
                                content=output.delta_text if output.delta_text else None,
                                tool_calls=output.delta_tool_calls,
                                reasoning_content=output.delta_reasoning_content,
                            )
                        elif tool_parser:
                            current_token_ids = output.outputs[0].token_ids if output.outputs else []
                            delta_token_ids = (
                                current_token_ids[len(previous_token_ids) :] if previous_token_ids else current_token_ids
                            )

                            raw_delta_message = self.extract_tool_calls_streaming(
                                model_name=request.model,
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request,
                            )
                            delta_message = self._coerce_stream_delta_message(
                                raw_delta_message,
                                fallback_text=delta_text,
                                default_role="assistant",
                            )
                            previous_text = current_text
                            previous_token_ids = current_token_ids

                            if delta_message is None and request.tools:
                                continue
                            if delta_message is None:
                                delta_message = DeltaMessage(content=delta_text, role="assistant")
                        else:
                            previous_text = current_text
                            current_token_ids = output.outputs[0].token_ids if output.outputs else []
                            previous_token_ids = current_token_ids
                            delta_message = DeltaMessage(content=delta_text, role="assistant")

                        delta_message = self._coerce_stream_delta_message(
                            delta_message,
                            fallback_text=delta_text,
                            default_role="assistant",
                        )
                        if delta_message is None:
                            continue

                        if delta_message and delta_message.tool_calls:
                            saw_tool_call_delta = True
                        if delta_message and request.tools:
                            content_text = delta_message.content if isinstance(delta_message.content, str) else None
                            if self._looks_like_tool_protocol_text(content_text):
                                delta_message.content = None
                        if saw_tool_call_delta and delta_message:
                            delta_message.content = None

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
                        total_generated = current_completion_tokens
                        generation_time = elapsed_time
                        tokens_per_second = current_tps

                    if disconnected:
                        return

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
                                finish_reason="tool_calls" if saw_tool_call_delta else "stop",
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
                    debug_context = self._build_stream_debug_context(
                        endpoint="/v1/chat/completions",
                        request_id=request_id,
                        model=request.model,
                        queue_kind=locals().get("kind"),
                        disconnected=locals().get("disconnected"),
                        output=locals().get("output"),
                        last_output=locals().get("last_output"),
                        previous_text=locals().get("previous_text"),
                        current_text=locals().get("current_text"),
                        delta_text=locals().get("delta_text"),
                        previous_token_ids=locals().get("previous_token_ids"),
                        current_token_ids=locals().get("current_token_ids"),
                        delta_token_ids=locals().get("delta_token_ids"),
                        raw_delta_message=locals().get("raw_delta_message"),
                        delta_message=locals().get("delta_message"),
                        saw_tool_call_delta=locals().get("saw_tool_call_delta"),
                        stream_error=locals().get("stream_error"),
                        tools=tools,
                        messages=messages,
                    )
                    logger.exception("Error during /v1/chat/completions streaming: %s | context=%s", e, debug_context)
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
                queue = self._start_stream_task(lambda: esurge.stream(prompt, sampling_params, request_id=request_id))
                total_generated = 0
                generation_time = 0.0
                tokens_per_second = 0.0
                last_output: RequestOutput | None = None
                disconnected = False

                try:
                    stream_error: Exception | None = None  # pyright: ignore[reportUnusedVariable]
                    while True:
                        if await raw_request.is_disconnected():
                            try:
                                esurge.abort_request(request_id)
                            except Exception:
                                logger.debug("Failed to abort request %s after disconnect", request_id, exc_info=True)
                            disconnected = True
                            break
                        kind, payload = await queue.get()
                        if kind == _STREAM_END:
                            break
                        if kind == _STREAM_ERROR:
                            stream_error = tp.cast(Exception, payload)  # pyright: ignore[reportUnusedVariable]
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

                    if disconnected:
                        return

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
                    debug_context = self._build_stream_debug_context(
                        endpoint="/v1/completions",
                        request_id=request_id,
                        model=request.model,
                        queue_kind=locals().get("kind"),
                        disconnected=locals().get("disconnected"),
                        output=locals().get("output"),
                        last_output=locals().get("last_output"),
                        previous_text=locals().get("previous_text"),
                        current_text=locals().get("current_text"),
                        delta_text=locals().get("delta_text"),
                        stream_error=locals().get("stream_error"),
                    )
                    logger.exception("Error during /v1/completions streaming: %s | context=%s", e, debug_context)
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
        """Create a standard chat completion response without function calling.

        Builds a ChatCompletionResponse from generation output, calculating
        usage statistics and formatting the response according to OpenAI API
        specifications.

        Args:
            request: The original chat completion request.
            output: The generation output containing completions.
            prompt_tokens: Number of tokens in the input prompt.
            start_time: Unix timestamp when generation started, used to
                calculate processing time.

        Returns:
            ChatCompletionResponse with a single choice containing the
            generated message and computed usage statistics.
        """
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
        """Define all API endpoints including admin authentication endpoints.

        Assembles the complete list of API endpoints by combining base server
        endpoints with admin authentication endpoints for API key management.

        The admin endpoints provide comprehensive key lifecycle management:
        - Key CRUD operations (create, read, update, delete)
        - Key lifecycle actions (suspend, reactivate, revoke, rotate)
        - Usage statistics and audit logging

        Returns:
            List of EndpointConfig objects defining all server endpoints,
            including both inference endpoints and admin management endpoints.

        Note:
            Admin endpoints require an API key with admin role for access.
            They are all prefixed with /v1/admin/.
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
