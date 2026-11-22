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

"""Enhanced FastAPI server that proxies requests to OpenAI API.

This module provides a proxy server that forwards requests to OpenAI's API
while adding EasyDeL-specific monitoring and compatibility features.
It enables seamless integration between EasyDeL inference engines and
OpenAI-compatible clients.

Classes:
    InferenceApiRouter: Main proxy server class with OpenAI API compatibility.
    ServerStatus: Enum for server operational states.
    ServerMetrics: Performance metrics tracking.
    EndpointConfig: API endpoint configuration.
    ErrorResponse: Standardized error response format.

Example:
    Proxy to OpenAI API::

        >>> from easydel.inference import InferenceApiRouter
        >>> router = InferenceApiRouter(
        ...     api_key="your-api-key",
        ...     base_url="https://api.openai.com/v1"
        ... )
        >>> router.run(host="0.0.0.0", port=8084)

    Proxy to a local EasyDeL server::

        >>> router = InferenceApiRouter(
        ...     base_url="http://localhost:8000/v1",
        ...     enable_function_calling=True
        ... )
        >>> router.run()
"""

from __future__ import annotations

import asyncio
import hmac
import inspect
import json
import os
import time
import typing as tp
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from http import HTTPStatus

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from easydel.workers.esurge.auth.credentials import api_key_from_headers

# The server scaffolding (status enum, metrics, endpoint/error models and the error-response helper) is shared with
# the eSurge API server; it is re-exported here so ``easydel.inference.oai_proxies.<name>`` keeps working.
from .inference_engine_interface import (
    TIMEOUT_KEEP_ALIVE,  # noqa: F401  (re-export)
    BaseInferenceApiServer,
    EndpointConfig,
    ErrorResponse,  # noqa: F401  (re-export)
    ServerMetrics,
    ServerStatus,
    create_error_response,
)
from .openai_api_modules import ChatCompletionRequest, CompletionRequest, ResponsesRequest, ResponsesToolChoice

_CHAT_SDK_PARAMS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "stream",
        "n",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "logit_bias",
        "user",
        "seed",
        "response_format",
        "stream_options",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
        "service_tier",
        "store",
        "metadata",
        "reasoning_effort",
        "modalities",
        "audio",
        "prediction",
        "web_search_options",
        "extra_headers",
        "extra_query",
        "timeout",
    }
)

_COMPLETION_SDK_PARAMS = frozenset(
    {
        "model",
        "prompt",
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "stream",
        "n",
        "logit_bias",
        "user",
        "seed",
        "stream_options",
        "logprobs",
        "echo",
        "best_of",
        "suffix",
        "extra_headers",
        "extra_query",
        "timeout",
    }
)

# ``ResponsesRequest`` fields that the OpenAI SDK's ``responses.create`` accepts as keyword arguments. Everything else
# (EasyDeL sampling extensions such as ``top_k``/``min_p``, chat-style ``messages``, legacy ``functions``, and any
# arbitrary extra field) is forwarded through ``extra_body`` instead of raising ``TypeError`` inside the SDK.
_RESPONSES_SDK_PARAMS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "store",
        "previous_response_id",
        "max_output_tokens",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "stream",
        "truncation",
        "include",
        "metadata",
        "extra_headers",
        "extra_query",
        "timeout",
    }
)


def _backend_error_status(error: BaseException) -> int:
    """Map an ``openai.APIError`` to the HTTP status the proxy should answer with.

    Only ``APIStatusError`` subclasses carry ``status_code``; connection and timeout errors do not, so reading the
    attribute unconditionally would raise ``AttributeError`` from inside the handler.

    Args:
        error: The exception raised by the OpenAI client.

    Returns:
        The backend's HTTP status when it produced one, ``504`` for timeouts and ``502`` for any other
        backend failure (for example an unreachable server).
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    if "timeout" in type(error).__name__.lower():
        return HTTPStatus.GATEWAY_TIMEOUT.value
    return HTTPStatus.BAD_GATEWAY.value


async def _close_upstream_stream(stream: tp.Any) -> None:
    """Close an OpenAI SDK ``AsyncStream`` or async generator cleanly."""
    close_fn = getattr(stream, "close", None)
    if callable(close_fn):
        try:
            result = close_fn()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass
        return
    aclose_fn = getattr(stream, "aclose", None)
    if callable(aclose_fn):
        try:
            await aclose_fn()
        except Exception:
            pass


class _CloseOnce:
    """Idempotent single-close guard for an upstream async stream."""

    __slots__ = ("_closed", "_stream")

    def __init__(self, stream: tp.Any) -> None:
        self._stream = stream
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await _close_upstream_stream(self._stream)


class _ManagedStreamBody:
    """Async iterator wrapper that guarantees shielded upstream stream closure even before first ``__anext__``."""

    def __init__(self, body_iter: AsyncGenerator[bytes, None], upstream_stream: tp.Any) -> None:
        self._body_iter = body_iter
        self._upstream_closer = (
            upstream_stream if isinstance(upstream_stream, _CloseOnce) else _CloseOnce(upstream_stream)
        )
        self._closed = False

    def __aiter__(self) -> _ManagedStreamBody:
        return self

    async def __anext__(self) -> bytes:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._body_iter.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Shield teardown against Starlette's task-group cancellation on client disconnect
        # so the body generator and upstream httpx/OpenAI AsyncStream always close cleanly.
        with anyio.CancelScope(shield=True):
            try:
                await self._body_iter.aclose()
            finally:
                await self._upstream_closer.close()


class _ManagedStreamingResponse(StreamingResponse):
    """StreamingResponse that always closes ``body_iterator`` on disconnect or send failure."""

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if callable(aclose):
                with anyio.CancelScope(shield=True):
                    await aclose()


def _serialize_oai_value(value: tp.Any) -> tp.Any:
    """Recursively convert nested Pydantic models into plain JSON-safe dicts without ``None`` fields."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_serialize_oai_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_oai_value(v) for k, v in value.items() if v is not None}
    return value


class InferenceApiRouter:
    """Enhanced FastAPI server acting as an OpenAI API proxy.

    This server provides a complete OpenAI API-compatible interface that can
    forward requests to either OpenAI's API or a local EasyDeL inference server.
    It includes additional monitoring, health check, and function calling endpoints.

    The router automatically detects backend capabilities and provides appropriate
    fallbacks when features are not available.

    Attributes:
        client: AsyncOpenAI client for backend communication.
        app: FastAPI application instance.
        status: Current server status (ServerStatus enum).
        metrics: Performance metrics tracker (ServerMetrics).
        base_url: Backend API base URL.
        enable_function_calling: Whether function calling is enabled.

    Example:
        >>> router = InferenceApiRouter(
        ...     api_key="sk-...",
        ...     base_url="https://api.openai.com/v1"
        ... )
        >>> router.run(host="0.0.0.0", port=8084)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        enable_function_calling: bool = True,
        http_timeout: float = 30.0,
        require_api_key: bool = False,
        proxy_api_key: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Inference API Router with EasyDeL compatibility.

        Args:
            api_key: OpenAI API key for backend authentication. If not provided,
                reads from OPENAI_API_KEY environment variable.
            base_url: Base URL for the backend API. If None, uses OpenAI's
                default endpoint.
            organization: OpenAI organization ID for billing attribution.
            enable_function_calling: Whether to enable function calling
                support and related endpoints. Defaults to True.
            http_timeout: Total timeout in seconds for auxiliary backend HTTP
                requests (``/metrics``, ``/v1/tools*``). Defaults to 30.0.
            require_api_key: When ``True``, require incoming requests to present
                a valid ``Authorization: Bearer`` or ``X-API-Key`` matching
                ``proxy_api_key`` (or ``api_key``). Defaults to ``False``.
            proxy_api_key: Optional client-facing API key when ``require_api_key=True``.
                Defaults to ``api_key`` when omitted.
            **kwargs: Additional arguments passed to AsyncOpenAI client.

        Raises:
            ImportError: If the openai package is not installed.
        """
        import openai  # type:ignore
        from openai import AsyncOpenAI  # type:ignore

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            organization=organization,
            **kwargs,
        )
        self.openai_module = openai
        self._api_key = resolved_api_key
        self._http_timeout = max(0.1, float(http_timeout))
        self._require_api_key = bool(require_api_key)
        self._proxy_api_key = proxy_api_key or resolved_api_key
        self._http_session: tp.Any | None = None
        self._http_session_loop: asyncio.AbstractEventLoop | None = None
        self._http_session_cls: tp.Any | None = None
        self._http_session_lock: asyncio.Lock | None = None
        self._http_session_lock_loop: asyncio.AbstractEventLoop | None = None
        self.enable_function_calling = enable_function_calling
        self.status = ServerStatus.STARTING
        self.metrics = ServerMetrics()
        self.base_url = str(base_url) if base_url else None

        self.app = FastAPI(
            title="EasyDeL Inference API Hub",
            description="High-performance inference server with OpenAI API compatibility",
            version="2.0.0",
            lifespan=self._lifespan,
        )

        self._register_endpoints()
        if enable_function_calling:
            self._add_function_calling_endpoints()

        self.status = ServerStatus.READY

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Manage proxy HTTP/OpenAI client lifecycle cleanly across server startup and shutdown."""
        self.status = ServerStatus.READY
        try:
            yield
        finally:
            self.status = ServerStatus.SHUTTING_DOWN
            await self.aclose()

    def _get_http_session_lock(self, loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
        """Return a per-event-loop lock serializing shared ``aiohttp.ClientSession`` lifecycle transitions."""
        lock = getattr(self, "_http_session_lock", None)
        lock_loop = getattr(self, "_http_session_lock_loop", None)
        if lock is None or lock_loop is not loop:
            lock = asyncio.Lock()
            self._http_session_lock = lock
            self._http_session_lock_loop = loop
        return lock

    async def aclose(self) -> None:
        """Close the shared backend HTTP session and OpenAI SDK client."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        async def _close_http_session() -> None:
            session = getattr(self, "_http_session", None)
            self._http_session = None
            self._http_session_loop = None
            self._http_session_cls = None
            if session is not None and not getattr(session, "closed", False):
                close_fn = getattr(session, "close", None)
                if callable(close_fn):
                    try:
                        result = close_fn()
                        if inspect.isawaitable(result):
                            await result
                    except Exception:
                        pass

        if loop is not None:
            async with self._get_http_session_lock(loop):
                await _close_http_session()
        else:
            await _close_http_session()

        client = getattr(self, "client", None)
        if client is not None:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    result = close_fn()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass

    def _presented_proxy_key(self, raw_request: Request | None) -> str | None:
        req_headers = getattr(raw_request, "headers", None) if raw_request is not None else None
        return api_key_from_headers(req_headers) if req_headers is not None else None

    def _check_proxy_auth(self, raw_request: Request | None) -> None:
        """Enforce proxy-level API key authentication when ``require_api_key=True``."""
        if not getattr(self, "_require_api_key", False):
            return
        expected = getattr(self, "_proxy_api_key", None) or getattr(self, "_api_key", None)
        presented = self._presented_proxy_key(raw_request)
        if not expected or not presented or not hmac.compare_digest(presented, str(expected)):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    def _backend_auth_headers(self, raw_request: Request | None = None) -> dict[str, str]:
        """Build backend authorization headers from the proxy's configured upstream API key.

        Client-presented proxy credentials (``proxy_api_key``) are never forwarded to the upstream backend.
        """
        del raw_request
        api_key = getattr(self, "_api_key", None) or getattr(getattr(self, "client", None), "api_key", None)
        if isinstance(api_key, str) and api_key.strip():
            return {"Authorization": f"Bearer {api_key.strip()}"}
        return {}

    async def _get_or_create_http_session(
        self,
        client_session_cls: tp.Any,
        client_timeout_cls: tp.Any,
    ) -> tp.Any:
        """Return a live shared ``aiohttp.ClientSession`` for the running loop, serializing init/replacement."""
        loop = asyncio.get_running_loop()
        session = getattr(self, "_http_session", None)
        if (
            session is not None
            and not getattr(session, "closed", False)
            and getattr(self, "_http_session_loop", None) is loop
            and getattr(self, "_http_session_cls", None) is client_session_cls
        ):
            return session

        lock = self._get_http_session_lock(loop)
        async with lock:
            session = getattr(self, "_http_session", None)
            if (
                session is not None
                and not getattr(session, "closed", False)
                and getattr(self, "_http_session_loop", None) is loop
                and getattr(self, "_http_session_cls", None) is client_session_cls
            ):
                return session
            old_session = session
            self._http_session = None
            self._http_session_loop = None
            self._http_session_cls = None
            if old_session is not None and not getattr(old_session, "closed", False):
                close_fn = getattr(old_session, "close", None)
                if callable(close_fn):
                    try:
                        res = close_fn()
                        if inspect.isawaitable(res):
                            await res
                    except Exception:
                        pass
            timeout = client_timeout_cls(total=float(getattr(self, "_http_timeout", 30.0)))
            new_session = client_session_cls(timeout=timeout)
            self._http_session = new_session
            self._http_session_loop = loop
            self._http_session_cls = client_session_cls
            return new_session

    async def _request_backend_http(
        self,
        method: str,
        url: str,
        *,
        raw_request: Request | None = None,
        json_body: tp.Any = None,
    ) -> tuple[int, tp.Any, bool]:
        """Execute a timed backend HTTP request reusing a shared ``aiohttp.ClientSession`` when available."""
        import aiohttp

        headers = self._backend_auth_headers(raw_request)
        client_session_cls = aiohttp.ClientSession
        client_timeout_cls = getattr(aiohttp, "ClientTimeout", None)

        async def _send_with_session(session: tp.Any) -> tuple[int, tp.Any, bool]:
            req_fn = session.get if method == "GET" else session.post
            req_kwargs: dict[str, tp.Any] = {}
            if method != "GET":
                req_kwargs["json"] = json_body
            if headers:
                try:
                    sig_params = inspect.signature(req_fn).parameters
                    if "headers" in sig_params or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
                    ):
                        req_kwargs["headers"] = headers
                except (TypeError, ValueError):
                    req_kwargs["headers"] = headers

            async with req_fn(url, **req_kwargs) as response:
                status = int(response.status)
                try:
                    return status, await response.json(), True
                except Exception:
                    text_fn = getattr(response, "text", None)
                    text = await text_fn() if callable(text_fn) else ""
                    return status, text, False

        if client_timeout_cls is not None:
            session = await self._get_or_create_http_session(client_session_cls, client_timeout_cls)
            return await _send_with_session(session)

        async with client_session_cls() as session:
            return await _send_with_session(session)

    @property
    def _endpoints(self) -> list[EndpointConfig]:
        """Define all API endpoints matching EasyDeL API servers.

        Returns:
            List of EndpointConfig objects defining the API surface.
        """
        return [
            EndpointConfig(
                path="/v1/responses",
                handler=self.responses,
                methods=["POST"],
                tags=["Responses"],
                summary="Create a response (new OpenAI Responses API)",
            ),
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
        """Add function calling specific endpoints to the router.

        Registers /v1/tools and /v1/tools/execute endpoints when
        function calling is enabled.
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
        """Register all API endpoints with the FastAPI application.

        Iterates through the endpoint configurations and adds each
        route to the FastAPI app.
        """
        BaseInferenceApiServer._register_endpoints(self)

    @staticmethod
    def _add_repetition_penalty(params: dict[str, tp.Any], repetition_penalty: float | None) -> None:
        """Forward ``repetition_penalty`` through ``extra_body`` only when it is non-default.

        ``None`` and the neutral value ``1.0`` are omitted so strict OpenAI-compatible
        upstreams that reject unknown body fields never see the EasyDeL-specific key
        unless the caller actually asked for a penalty.

        Args:
            params: OpenAI SDK keyword arguments, mutated in place.
            repetition_penalty: Requested multiplicative repetition penalty.
        """
        if repetition_penalty is None or repetition_penalty == 1.0:
            return
        params["extra_body"] = {"repetition_penalty": repetition_penalty}

    @staticmethod
    def _server_root(base_url: str | None) -> str | None:
        """Return the server root for a client ``base_url`` that may end in ``/v1``.

        Args:
            base_url: OpenAI-client base URL (for example ``http://localhost:8000/v1``) or ``None``.

        Returns:
            ``base_url`` without trailing slashes and without a trailing ``/v1`` segment, or ``None``.
        """
        if not base_url:
            return None
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return root.rstrip("/")

    @property
    def server_root_url(self) -> str | None:
        """Server root used for the non-OpenAI endpoints (``/metrics``, ``/v1/tools*``).

        EasyDeL serves those endpoints at the server root, while ``base_url`` is the OpenAI-client base URL that
        normally ends in ``/v1``; the suffix is stripped so the proxied URLs resolve correctly.

        Returns:
            The backend root URL, or ``None`` when no ``base_url`` was configured.
        """
        return self._server_root(getattr(self, "base_url", None))

    def build_oai_params_from_responses_request(self, request: ResponsesRequest) -> dict[str, tp.Any]:
        """Build ``client.responses.create`` keyword arguments from a Responses request.

        Fields the OpenAI SDK accepts are passed as keyword arguments. EasyDeL-only fields
        (``top_k``, ``min_p``, ``repetition_penalty`` ...) and arbitrary extras are forwarded through
        ``extra_body`` so the SDK does not raise ``TypeError``. Chat-style ``messages`` are used as ``input``
        when no ``input`` is given, and ``max_tokens`` / ``max_completion_tokens`` fill ``max_output_tokens``.

        Args:
            request: The Responses API request to convert.

        Returns:
            Dictionary of keyword arguments for ``responses.create``.
        """
        dumped = request.model_dump(exclude_none=True)

        if dumped.get("input") is None and dumped.get("messages"):
            dumped["input"] = dumped["messages"]
        if dumped.get("max_output_tokens") is None:
            for alias in ("max_completion_tokens", "max_tokens"):
                if dumped.get(alias) is not None:
                    dumped["max_output_tokens"] = dumped[alias]
                    break
        if dumped.get("repetition_penalty") == 1.0:
            dumped.pop("repetition_penalty")

        params: dict[str, tp.Any] = {}
        extra_body: dict[str, tp.Any] = dict(dumped.pop("extra_body", None) or {})
        for key, value in dumped.items():
            if key in ("messages", "max_tokens", "max_completion_tokens"):
                continue  # already folded into ``input`` / ``max_output_tokens``
            if key == "reasoning" and not isinstance(value, dict):
                extra_body[key] = value  # bare boolean is an EasyDeL extension; the SDK expects a config object
            elif key in _RESPONSES_SDK_PARAMS:
                params[key] = value
            else:
                extra_body[key] = value
        if extra_body:
            params["extra_body"] = extra_body
        return params

    @staticmethod
    def _merge_extensions_and_extras(
        params: dict[str, tp.Any],
        request: ChatCompletionRequest | CompletionRequest,
        sdk_params: frozenset[str],
    ) -> None:
        """Forward standard extra OpenAI SDK params directly and non-default EasyDeL extensions via ``extra_body``."""
        extra_body: dict[str, tp.Any] = {}
        existing_extra_body = params.get("extra_body")
        if isinstance(existing_extra_body, dict):
            extra_body.update(existing_extra_body)
        req_extra_body = getattr(request, "extra_body", None)
        if isinstance(req_extra_body, dict):
            extra_body.update(_serialize_oai_value(req_extra_body))

        rep_penalty = getattr(request, "repetition_penalty", None)
        if rep_penalty is not None and rep_penalty != 1.0:
            extra_body["repetition_penalty"] = rep_penalty

        top_k = getattr(request, "top_k", None)
        if top_k is not None and top_k != 0:
            extra_body["top_k"] = top_k

        min_p = getattr(request, "min_p", None)
        if min_p is not None and min_p != 0.0:
            extra_body["min_p"] = min_p

        if bool(getattr(request, "ignore_eos", False)):
            extra_body["ignore_eos"] = True

        suppress_tokens = getattr(request, "suppress_tokens", None)
        if suppress_tokens:
            extra_body["suppress_tokens"] = list(suppress_tokens)

        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)
        if chat_template_kwargs is not None:
            extra_body["chat_template_kwargs"] = _serialize_oai_value(chat_template_kwargs)

        model_extra = getattr(request, "model_extra", None)
        if isinstance(model_extra, dict):
            for key, value in model_extra.items():
                if value is None or key in ("extra_body", "request_id"):
                    continue
                serialized = _serialize_oai_value(value)
                if key in sdk_params:
                    params[key] = serialized
                else:
                    extra_body[key] = serialized

        if extra_body:
            params["extra_body"] = extra_body

    def build_oai_params_from_request(
        self,
        request: CompletionRequest,
    ) -> dict[str, tp.Any]:
        """Build OpenAI parameters from a completion request.

        Converts a CompletionRequest object into a dictionary of parameters
        suitable for the OpenAI API completions endpoint. Non-default EasyDeL
        extensions (``repetition_penalty``, ``top_k``, ``min_p``, ``ignore_eos``,
        ``suppress_tokens``, etc.) are forwarded through ``extra_body``.

        Args:
            request: The completion request to convert.

        Returns:
            Dictionary of OpenAI API parameters ready for the completions endpoint.
        """
        params: dict[str, tp.Any] = {
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
        if request.logit_bias is not None:
            params["logit_bias"] = dict(request.logit_bias)
        if request.user is not None:
            params["user"] = request.user
        self._merge_extensions_and_extras(params, request, _COMPLETION_SDK_PARAMS)
        return params

    def build_oai_params_from_chat_request(
        self,
        request: ChatCompletionRequest,
    ) -> dict[str, tp.Any]:
        """Build OpenAI parameters from a chat completion request.

        Converts a ChatCompletionRequest object into a dictionary of parameters
        suitable for the OpenAI API chat completions endpoint, dumping nested
        Pydantic models with ``exclude_none=True`` and forwarding non-default
        EasyDeL extensions through ``extra_body``.

        Args:
            request: The chat completion request to convert.

        Returns:
            Dictionary of OpenAI API parameters with optional tool/function
            definitions for the chat completions endpoint.
        """
        params: dict[str, tp.Any] = {
            "model": request.model,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop,
            "stream": request.stream,
            "n": request.n,
        }
        max_completion_tokens = getattr(request, "max_completion_tokens", None)
        if max_completion_tokens is not None:
            params["max_completion_tokens"] = max_completion_tokens
        if request.logit_bias is not None:
            params["logit_bias"] = dict(request.logit_bias)
        if request.user is not None:
            params["user"] = request.user

        if request.tools:
            params["tools"] = [tool.model_dump(exclude_none=True) for tool in request.tools]
        if request.tool_choice is not None:
            if isinstance(request.tool_choice, ResponsesToolChoice):
                params["tool_choice"] = {
                    "type": request.tool_choice.type,
                    "function": {"name": request.tool_choice.name},
                }
            elif hasattr(request.tool_choice, "model_dump"):
                params["tool_choice"] = request.tool_choice.model_dump(exclude_none=True)
            else:
                params["tool_choice"] = request.tool_choice
        if request.functions:
            params["functions"] = [func.model_dump(exclude_none=True) for func in request.functions]
        if request.function_call is not None:
            if hasattr(request.function_call, "model_dump"):
                params["function_call"] = request.function_call.model_dump(exclude_none=True)
            else:
                params["function_call"] = request.function_call

        self._merge_extensions_and_extras(params, request, _CHAT_SDK_PARAMS)
        return params

    def process_request_params(
        self,
        openai_params: dict,
        request: ChatCompletionRequest | CompletionRequest,
    ) -> tuple[dict, BaseModel | None]:
        """Process request parameters before sending to OpenAI.

        Hook method for subclasses to modify parameters or extract metadata
        before forwarding the request to the backend API.

        Args:
            openai_params: Dictionary of OpenAI API parameters.
            request: Original request object containing the full request data.

        Returns:
            Tuple of (processed_params, optional_metadata) where processed_params
            is the potentially modified parameter dictionary and optional_metadata
            is any metadata to include in the response stream.
        """
        return openai_params, None

    @staticmethod
    def _format_metadata_sse(metadata: BaseModel | dict[str, tp.Any] | tp.Any) -> bytes:
        """Format metadata as a standard SSE event payload.

        Serializes ``metadata`` (Pydantic model, dict, or arbitrary value)
        into a UTF-8 encoded ``event: metadata\\ndata: ...\\n\\n`` frame for
        Server-Sent Event streaming.

        Args:
            metadata: The metadata payload to encode.

        Returns:
            The encoded SSE frame as bytes ready to write to the client.
        """
        if hasattr(metadata, "model_dump_json"):
            payload = metadata.model_dump_json(exclude_unset=True)
        elif isinstance(metadata, dict):
            payload = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)
        else:
            payload = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False, default=str)
        return f"event: metadata\ndata: {payload}\n\n".encode()

    async def responses(self, request: ResponsesRequest, raw_request: Request) -> tp.Any:
        """Handle OpenAI Responses API requests.

        Proxies POST /v1/responses to the configured backend using the
        official OpenAI client. This enables clients that use the new
        Responses API to work with either OpenAI or an OpenAI-compatible
        EasyDeL server.

        Args:
            request: The Responses API request containing model, input,
                and generation parameters.
            raw_request: Raw FastAPI request containing headers and metadata.

        Returns:
            Response object or StreamingResponse for streamed responses.

        Raises:
            HTTPException: If the backend API returns an error.
        """
        request_id = raw_request.headers.get("X-Request-ID")

        try:
            self._check_proxy_auth(raw_request)
            self.metrics.total_requests += 1
            params = self.build_oai_params_from_responses_request(request)
            if not bool(params.get("stream", False)):
                response = await self.client.responses.create(**params)
                self.metrics.successful_requests += 1
                return response

            # Open the upstream stream before returning StreamingResponse so upstream
            # 4xx/5xx APIError propagates as its real HTTP status rather than HTTP 200.
            stream = await self.client.responses.create(**params)
            closer = _CloseOnce(stream)

            async def stream_events() -> AsyncGenerator[bytes, None]:
                """Stream Server-Sent Events from the backend."""
                try:
                    async for event in stream:
                        event_type = getattr(event, "type", None) or getattr(event, "event", None)
                        if hasattr(event, "model_dump_json"):
                            data = event.model_dump_json(exclude_unset=True)
                        else:
                            data = json.dumps(event)
                        if event_type:
                            yield f"event: {event_type}\ndata: {data}\n\n".encode()
                        else:
                            yield f"data: {data}\n\n".encode()
                    self.metrics.successful_requests += 1
                except Exception as e:
                    self.metrics.failed_requests += 1
                    error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)
                    yield f"data: {error_response.body.decode()}\n\n".encode()
                finally:
                    with anyio.CancelScope(shield=True):
                        await closer.close()

            return _ManagedStreamingResponse(
                _ManagedStreamBody(stream_events(), closer),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id or "",
                },
            )

        except self.openai_module.APIError as e:
            self.metrics.failed_requests += 1
            raise HTTPException(status_code=_backend_error_status(e), detail=str(e)) from e
        except HTTPException:
            self.metrics.failed_requests += 1
            raise
        except Exception as e:
            self.metrics.failed_requests += 1
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def chat_completions(self, request: ChatCompletionRequest, raw_request: Request = None) -> tp.Any:
        """Handle chat completion requests with function calling support.

        Processes POST /v1/chat/completions requests, forwarding them to
        the backend API and returning either a complete response or a
        streaming response based on the request parameters.

        Args:
            request: The chat completion request containing messages,
                model selection, and generation parameters.
            raw_request: Optional raw FastAPI request for auth and request ID headers.

        Returns:
            Chat completion response object or StreamingResponse for
            streamed responses.

        Raises:
            HTTPException: If the backend API returns an error.
        """
        request_id = getattr(request, "request_id", None) or (
            raw_request.headers.get("X-Request-ID") if raw_request is not None else None
        )

        try:
            self._check_proxy_auth(raw_request)
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
                stream = await self.client.chat.completions.create(**openai_params)
                closer = _CloseOnce(stream)
                return _ManagedStreamingResponse(
                    _ManagedStreamBody(
                        self._stream_chat_completion(stream, metadata, request_id, _closer=closer),
                        closer,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Request-ID": request_id or "",
                    },
                )

        except self.openai_module.APIError as e:
            self.metrics.failed_requests += 1
            raise HTTPException(status_code=_backend_error_status(e), detail=str(e)) from e
        except HTTPException:
            self.metrics.failed_requests += 1
            raise
        except Exception as e:
            self.metrics.failed_requests += 1
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _stream_chat_completion(
        self,
        stream_or_params: tp.Any,
        metadata: dict | None,
        request_id: str | None = None,
        *,
        _closer: _CloseOnce | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Handle streaming chat completion responses.

        Streams Server-Sent Events (SSE) formatted responses from the backend
        API to the client and always closes the upstream stream in ``finally``.

        Args:
            stream_or_params: Already-opened upstream async stream, or a dict of
                OpenAI API parameters for backward compatibility.
            metadata: Optional metadata to include at the start of the stream.
            request_id: Request identifier for tracking and logging.

        Yields:
            SSE-formatted bytes containing response chunks, followed by
            a [DONE] marker at the end.
        """
        stream: tp.Any = None
        closer = _closer
        try:
            stream = (
                await self.client.chat.completions.create(**stream_or_params)
                if isinstance(stream_or_params, dict)
                else stream_or_params
            )
            if closer is None and stream is not None:
                closer = _CloseOnce(stream)
            if metadata is not None:
                yield self._format_metadata_sse(metadata)

            async for chunk in stream:
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n".encode()

            yield b"data: [DONE]\n\n"
            self.metrics.successful_requests += 1

        except Exception as e:
            self.metrics.failed_requests += 1
            error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)
            yield f"data: {error_response.body.decode()}\n\n".encode()
        finally:
            if closer is not None:
                with anyio.CancelScope(shield=True):
                    await closer.close()

    async def completions(self, request: CompletionRequest, raw_request: Request = None) -> tp.Any:
        """Handle text completion requests.

        Processes POST /v1/completions requests, forwarding them to
        the backend API and returning either a complete response or a
        streaming response based on the request parameters.

        Args:
            request: The completion request containing prompt, model
                selection, and generation parameters.
            raw_request: Optional raw FastAPI request for auth and request ID headers.

        Returns:
            Completion response object or StreamingResponse for
            streamed responses.

        Raises:
            HTTPException: If the backend API returns an error.
        """
        request_id = getattr(request, "request_id", None) or (
            raw_request.headers.get("X-Request-ID") if raw_request is not None else None
        )

        try:
            self._check_proxy_auth(raw_request)
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
                stream = await self.client.completions.create(**openai_params)
                closer = _CloseOnce(stream)
                return _ManagedStreamingResponse(
                    _ManagedStreamBody(
                        self._stream_completion(stream, metadata, request_id, _closer=closer),
                        closer,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Request-ID": request_id or "",
                    },
                )

        except self.openai_module.APIError as e:
            self.metrics.failed_requests += 1
            raise HTTPException(status_code=_backend_error_status(e), detail=str(e)) from e
        except HTTPException:
            self.metrics.failed_requests += 1
            raise
        except Exception as e:
            self.metrics.failed_requests += 1
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _stream_completion(
        self,
        stream_or_params: tp.Any,
        metadata: dict | None,
        request_id: str | None = None,
        *,
        _closer: _CloseOnce | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Handle streaming completion responses.

        Streams Server-Sent Events (SSE) formatted responses from the backend
        API to the client and always closes the upstream stream in ``finally``.

        Args:
            stream_or_params: Already-opened upstream async stream, or a dict of
                OpenAI API parameters for backward compatibility.
            metadata: Optional metadata to include at the start of the stream.
            request_id: Request identifier for tracking and logging.

        Yields:
            SSE-formatted bytes containing response chunks, followed by
            a [DONE] marker at the end.
        """
        stream: tp.Any = None
        closer = _closer
        try:
            stream = (
                await self.client.completions.create(**stream_or_params)
                if isinstance(stream_or_params, dict)
                else stream_or_params
            )
            if closer is None and stream is not None:
                closer = _CloseOnce(stream)
            if metadata is not None:
                yield self._format_metadata_sse(metadata)

            async for chunk in stream:
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n".encode()

            yield b"data: [DONE]\n\n"
            self.metrics.successful_requests += 1

        except Exception as e:
            self.metrics.failed_requests += 1
            error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)
            yield f"data: {error_response.body.decode()}\n\n".encode()
        finally:
            if closer is not None:
                with anyio.CancelScope(shield=True):
                    await closer.close()

    async def health_check(self, raw_request: Request = None) -> JSONResponse:
        """Perform a comprehensive health check.

        Checks the backend API connectivity by listing models and returns
        detailed health status including uptime and request metrics.

        Returns:
            JSONResponse with health status, model count, backend URL,
            and request statistics. Returns 200 if healthy, 503 if unhealthy.
        """
        self._check_proxy_auth(raw_request)
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
        """Perform a liveness check.

        Simple endpoint to verify the server process is running.
        Used by orchestrators like Kubernetes for liveness probes.

        Returns:
            JSONResponse with status "alive" and HTTP 200.
        """
        return JSONResponse({"status": "alive"}, status_code=200)

    async def readiness(self, raw_request: Request = None) -> JSONResponse:
        """Perform a readiness check.

        Verifies the server is ready to receive traffic. When ``require_api_key=True``,
        unauthenticated probes only check local proxy readiness without triggering
        upstream backend requests or exposing backend error details; authenticated
        callers also probe upstream connectivity.

        Returns:
            JSONResponse with status "ready" and HTTP 200 if ready,
            otherwise status "not ready" with HTTP 503.
        """
        authenticated = False
        if getattr(self, "_require_api_key", False):
            presented = self._presented_proxy_key(raw_request)
            if presented is not None:
                self._check_proxy_auth(raw_request)
                authenticated = True
            else:
                if self.status == ServerStatus.READY:
                    return JSONResponse({"status": "ready"}, status_code=200)
                return JSONResponse({"status": "not ready"}, status_code=503)

        if self.status != ServerStatus.READY:
            return JSONResponse({"status": "not ready"}, status_code=503)
        try:
            await self.client.models.list()
            return JSONResponse({"status": "ready"}, status_code=200)
        except Exception as e:
            if authenticated:
                return JSONResponse({"status": "not ready", "error": str(e)}, status_code=503)
            return JSONResponse({"status": "not ready"}, status_code=503)

    async def get_metrics(self, raw_request: Request = None) -> JSONResponse:
        """Get server performance metrics.

        Returns detailed metrics about the server's operation including
        request counts, token generation stats, and backend metrics
        if available.

        Returns:
            JSONResponse containing api_router_metrics with uptime,
            request counts, token stats, and optionally backend_metrics
            if the backend supports the /metrics endpoint.
        """
        self._check_proxy_auth(raw_request)
        self.metrics.uptime_seconds = time.time() - self.metrics.start_time

        # If backend is an EasyDeL server, try to get its metrics
        backend_metrics = None
        if self.base_url:
            try:
                status, data, is_json = await self._request_backend_http(
                    "GET",
                    f"{self.server_root_url}/metrics",
                    raw_request=raw_request,
                )
                if status == 200 and is_json:
                    backend_metrics = data
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

    async def list_models(self, raw_request: Request = None) -> JSONResponse:
        """List available models with metadata.

        Retrieves the list of models from the backend API and enriches
        it with additional metadata about capabilities.

        Returns:
            JSONResponse containing a list of model objects with id,
            ownership, and capability metadata.

        Raises:
            Returns error response if backend API call fails.
        """
        self._check_proxy_auth(raw_request)
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

                if self.base_url:
                    model_info["metadata"] = {
                        "supports_chat": True,  # Assume true for now
                        "supports_function_calling": self.enable_function_calling,
                        "backend_type": "custom",
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

    async def get_model(self, model_id: str, raw_request: Request = None) -> JSONResponse:
        """Get detailed information about a specific model.

        Retrieves metadata for a single model from the backend API.

        Args:
            model_id: The unique identifier of the model to retrieve.

        Returns:
            JSONResponse containing model details including id, ownership,
            and capability metadata.

        Raises:
            Returns 404 error if model is not found.
            Returns 500 error for other backend failures.
        """
        self._check_proxy_auth(raw_request)
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

            if self.base_url:
                model_info["metadata"] = {
                    "supports_chat": True,
                    "supports_function_calling": self.enable_function_calling,
                    "backend_type": "custom",
                }

            return JSONResponse(model_info, status_code=200)

        except self.openai_module.NotFoundError:
            return create_error_response(HTTPStatus.NOT_FOUND, f"Model '{model_id}' not found")
        except Exception as e:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    async def list_tools(self, raw_request: Request = None) -> JSONResponse:
        """List available tools/functions for each model.

        Returns information about available tools and functions that
        can be used with function calling.

        Returns:
            JSONResponse containing tools information. If function calling
            is disabled, returns an empty tools list. If the backend
            supports the /v1/tools endpoint, proxies that response.
        """
        self._check_proxy_auth(raw_request)
        if not self.enable_function_calling:
            return JSONResponse(
                {
                    "message": "Function calling is disabled",
                    "tools": [],
                },
                status_code=200,
            )

        # If backend exposes tools, try to fetch them directly
        if self.base_url:
            try:
                status, data, is_json = await self._request_backend_http(
                    "GET",
                    f"{self.server_root_url}/v1/tools",
                    raw_request=raw_request,
                )
                if status == 200 and is_json:
                    return JSONResponse(data, status_code=200)
            except Exception:
                # Backend doesn't support tools endpoint
                pass

        return JSONResponse(
            {
                "models": {},
                "default_format": "openai",
                "message": "Backend does not provide tools information",
            },
            status_code=200,
        )

    async def execute_tool(self, request: Request) -> JSONResponse:
        """Execute a tool/function call.

        Proxies tool execution requests to the backend if it supports
        the /v1/tools/execute endpoint.

        Args:
            request: FastAPI Request containing the tool execution payload.

        Returns:
            JSONResponse with the tool execution result from the backend.

        Raises:
            HTTPException: 401 when ``require_api_key=True`` and credentials are invalid.
            Returns 501 error if function calling is disabled.
            Returns 400 error if request body is not valid JSON.
            Returns 502 error if backend returns a non-JSON reply.
            Returns 500 error if tool execution fails.
            Returns 501 error if backend doesn't support tool execution.
        """
        self._check_proxy_auth(request)
        if not self.enable_function_calling:
            return create_error_response(HTTPStatus.NOT_IMPLEMENTED, "Function calling is disabled")

        # If backend exposes tool execution, proxy the request
        if self.base_url:
            try:
                body = await request.json()
            except Exception as e:
                return create_error_response(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {e!s}")
            try:
                status, response_data, is_json = await self._request_backend_http(
                    "POST",
                    f"{self.server_root_url}/v1/tools/execute",
                    raw_request=request,
                    json_body=body,
                )
                if not is_json:
                    try:
                        http_status = HTTPStatus(status) if status >= 400 else HTTPStatus.BAD_GATEWAY
                    except ValueError:
                        http_status = HTTPStatus.BAD_GATEWAY
                    detail = str(response_data).strip() or f"Backend returned non-JSON response (status {status})"
                    return create_error_response(http_status, detail)
                return JSONResponse(response_data, status_code=status)
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
        """Start the server with enhanced configuration.

        Launches the uvicorn server with the specified configuration.
        Optionally uses uvloop for improved async performance if available.

        Args:
            host: Host address to bind to. Defaults to "0.0.0.0" (all interfaces).
            port: Port number to listen on. Defaults to 8084.
            log_level: Logging level (debug, info, warning, error). Defaults to "info".
            ssl_keyfile: Path to SSL private key file for HTTPS.
            ssl_certfile: Path to SSL certificate file for HTTPS.
            workers: Number of worker processes. Defaults to 1.
            reload: Enable auto-reload for development. Defaults to False.

        Note:
            When reload=True, workers is forced to 1.
            SSL requires both ssl_keyfile and ssl_certfile to be provided.
        """
        BaseInferenceApiServer.run(
            self,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            reload=reload,
        )

    fire = run
