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
"""Implements an improved FastAPI server for serving vSurge models with OpenAI API compatibility."""

from __future__ import annotations

import asyncio
import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import ProcessorMixin

from easydel.inference.utilities import SamplingParams
from easydel.inference.vsurge.utils._utils import ReturnSample
from easydel.utils.helpers import get_logger

from ...openai_api_modules import (
    ChatCompletionRequest,
    ChatCompletionRequestWithTools,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatMessage,
    ChatMessageWithTools,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
    CompletionStreamResponseChoice,
    DeltaMessage,
    FunctionCallFormat,
    FunctionCallFormatter,
    FunctionCallParser,
    ToolCall,
    UsageInfo,
)
from ..vsurge import vSurge

TIMEOUT_KEEP_ALIVE = 5.0
logger = get_logger("vSurgeApiServer")


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


class vSurgeApiServer:
    """
    Enhanced FastAPI server for serving vSurge instances with function calling.

    Features:
    - OpenAI API compatibility
    - Function/Tool calling support
    - Multiple function call formats (OpenAI, Hermes, Gorilla, JSON)
    - Streaming function calls
    - Parallel tool calls
    """

    def __init__(
        self,
        vsurge_map: dict[str, vSurge] | vSurge,
        max_workers: int | None = None,
        oai_like_processor: bool = True,
        enable_cors: bool = True,
        cors_origins: list[str] | None = None,
        max_request_size: int = 10 * 1024 * 1024,
        request_timeout: float = 300.0,
        enable_function_calling: bool = True,
        default_function_format: FunctionCallFormat = FunctionCallFormat.OPENAI,
    ) -> None:
        """
        Initialize the vSurge API Server with function calling support.

        Args:
            vsurge_map: Dictionary of model names to vSurge instances
            max_workers: Maximum number of worker threads
            oai_like_processor: Use OpenAI-like conversation format
            enable_cors: Enable CORS middleware
            cors_origins: Allowed CORS origins
            max_request_size: Maximum request size in bytes
            request_timeout: Request timeout in seconds
            enable_function_calling: Enable function calling support
            default_function_format: Default format for function calls
        """
        if isinstance(vsurge_map, vSurge):
            vsurge_map = {vsurge_map.vsurge_name: vsurge_map}

        self.vsurge_map: dict[str, vSurge] = {}
        self._validate_and_register_models(vsurge_map)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vsurge-worker")
        self.oai_like_processor = oai_like_processor
        self.max_request_size = max_request_size
        self.request_timeout = request_timeout
        self.status = ServerStatus.STARTING
        self.metrics = ServerMetrics()
        self._active_requests: set[str] = set()
        self._request_lock = asyncio.Lock()
        self.enable_function_calling = enable_function_calling
        self.default_function_format = default_function_format

        self.app = FastAPI(
            title="EasyDeL vSurge API Server",
            description="High-performance inference server with OpenAI API compatibility",
            version="2.0.0",
            lifespan=self._lifespan,
        )
        if enable_function_calling:
            self._add_function_calling_endpoints()

        if enable_cors:
            self._setup_cors(cors_origins)
        self._setup_middleware()
        self._register_endpoints()

        logger.info(f"vSurge API Server initialized with {len(self.vsurge_map)} models")

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Manage server lifecycle."""
        logger.info("Starting vSurge API Server...")
        self.status = ServerStatus.READY
        yield
        logger.info("Shutting down vSurge API Server...")
        self.status = ServerStatus.SHUTTING_DOWN
        await self._graceful_shutdown()

    def _validate_and_register_models(self, vsurge_map: dict[str, vSurge]) -> None:
        """Validate and register vSurge models."""
        for name, vsurge in vsurge_map.items():
            if not isinstance(vsurge, vSurge):
                raise TypeError(f"Value for key '{name}' must be an instance of vSurge, got {type(vsurge).__name__}")
            self.vsurge_map[name] = vsurge
            logger.info(f"Registered model: {name}")

    def _setup_cors(self, origins: list[str] | None) -> None:
        """Setup CORS middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_middleware(self) -> None:
        """Setup request middleware."""

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

    def _get_vsurge(self, model_name: str) -> vSurge:
        """Get vSurge instance by model name."""
        vsurge = self.vsurge_map.get(model_name)
        if vsurge is None:
            available = list(self.vsurge_map.keys())
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Available models: {available}")
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
            if request.chat_template_kwargs is None:
                request.chat_template_kwargs = {}
            add_generation_prompt = request.chat_template_kwargs.pop("add_generation_prompt", True)
            return processor.apply_chat_template(
                tokenize=False,
                conversation=conversation,
                add_generation_prompt=add_generation_prompt,
                **request.chat_template_kwargs,
            )
        except Exception as e:
            self.logger.exception(f"Error applying chat template for model {vsurge.vsurge_name}: {e}")
            raise RuntimeError(f"Error tokenizing input: {e}") from e

    async def _prepare_vsurge_input_async(self, request, vsurge) -> dict:
        """Runs tokenization in the thread pool."""
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self._prepare_vsurge_input,
            request,
            vsurge,
        )

    def _prepare_vsurge_input_with_tools(
        self,
        request: ChatCompletionRequestWithTools,
        vsurge: vSurge,
    ) -> str:
        """Prepare input with function/tool definitions."""
        messages = [msg.model_dump() for msg in request.messages]
        processor = vsurge.processor

        if isinstance(processor, ProcessorMixin) and self.oai_like_processor:
            from easydel.trainers.prompt_utils import convert_to_openai_format

            messages = convert_to_openai_format(messages)

        tools = request.get_tools()
        if tools:
            format_type = request.function_call_format or self.default_function_format

            tools_prompt = FunctionCallFormatter.format_tools_for_prompt(tools, format_type)
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += f"\n\n{tools_prompt}"
            else:
                messages.insert(0, {"role": "system", "content": tools_prompt})

        try:
            if request.chat_template_kwargs is None:
                request.chat_template_kwargs = {}
            add_generation_prompt = request.chat_template_kwargs.pop("add_generation_prompt", True)

            return processor.apply_chat_template(
                conversation=messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                **request.chat_template_kwargs,
            )
        except Exception as e:
            logger.exception(f"Error applying chat template: {e}")
            raise RuntimeError(f"Error preparing input: {e}") from e

    async def _prepare_vsurge_input_with_tools_async(
        self,
        request: ChatCompletionRequestWithTools,
        vsurge: vSurge,
    ) -> str:
        """Prepare input with tools asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._prepare_vsurge_input_with_tools,
            request,
            vsurge,
        )

    async def _handle_completion_with_tools_async(
        self,
        request: ChatCompletionRequest | ChatCompletionRequestWithTools,
        vsurge: vSurge,
        content: str,
        request_id: str | None = None,
        is_function_request: bool = False,
    ) -> ChatCompletionResponse:
        """Generate non-streaming response with function calling support."""
        start_time = time.time()

        try:
            prompt_tokens = vsurge.count_tokens(content)
            sampling_params = self._create_sampling_params(request)

            response = await vsurge.generate(prompts=content, sampling_params=sampling_params, stream=False)

            if response is None:
                raise RuntimeError("Generation failed to produce output")

            result: ReturnSample = response[0]

            completion_tokens = sum(result.num_generated_tokens)
            self.metrics.total_tokens_generated += completion_tokens
            generation_time = time.time() - start_time
            tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0

            if self.metrics.average_tokens_per_second == 0:
                self.metrics.average_tokens_per_second = tokens_per_second
            else:
                self.metrics.average_tokens_per_second = (
                    self.metrics.average_tokens_per_second * 0.9 + tokens_per_second * 0.1
                )

            choices = []
            for idx in range(len(result.text)):
                response_text = result.accumulated_text[idx]

                if is_function_request:
                    format_type = getattr(request, "function_call_format", self.default_function_format)
                    parser = FunctionCallParser(format=format_type, strict=False)
                    function_calls = parser.parse(response_text)

                    if function_calls:
                        message = ChatMessageWithTools.from_function_calls(function_calls, content=None)
                        finish_reason = "tool_calls"
                    else:
                        message = ChatMessage(role="assistant", content=response_text)
                        finish_reason = self._determine_finish_reason(
                            result.num_generated_tokens[idx], sampling_params.max_tokens, response_text
                        )
                else:
                    message = ChatMessage(role="assistant", content=response_text)
                    finish_reason = self._determine_finish_reason(
                        result.num_generated_tokens[idx], sampling_params.max_tokens, response_text
                    )

                choices.append(
                    ChatCompletionResponseChoice(
                        index=idx,
                        message=message,
                        finish_reason=finish_reason,
                    )
                )

            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                tokens_per_second=tokens_per_second,
                processing_time=generation_time,
            )

            return ChatCompletionResponse(
                model=request.model,
                choices=choices,
                usage=usage,
            )

        except Exception as e:
            logger.exception(f"Error generating response: {e}")
            raise

    async def _handle_streaming_with_tools_async(
        self,
        request: ChatCompletionRequest | ChatCompletionRequestWithTools,
        vsurge: vSurge,
        content: str,
        request_id: str | None = None,
        is_function_request: bool = False,
    ) -> StreamingResponse:
        """Generate streaming response with function calling support."""

        async def generate_stream():
            start_time = time.time()
            prompt_tokens = vsurge.count_tokens(content)
            sampling_params = self._create_sampling_params(request)

            parser = None
            format_type = None
            if is_function_request:
                format_type = getattr(request, "function_call_format", self.default_function_format)
                parser = FunctionCallParser(format=format_type, strict=False)

            try:
                total_generated = 0
                first_token_time = None
                accumulated_texts = {}
                partial_function_calls = {}
                function_call_indicators = {
                    FunctionCallFormat.OPENAI: ['{"name"', "Function call:", "```json"],
                    FunctionCallFormat.HERMES: ["<tool_call>"],
                    FunctionCallFormat.GORILLA: ["<<<"],
                    FunctionCallFormat.JSON_SCHEMA: ["{", '"function"'],
                }

                async for response_state in await vsurge.generate(
                    prompts=content,
                    sampling_params=sampling_params,
                    stream=True,
                ):
                    response_state: ReturnSample = response_state[0]

                    if first_token_time is None and response_state.num_generated_tokens[0] > 0:
                        first_token_time = time.time() - start_time
                    new_tokens = sum(response_state.num_generated_tokens) - total_generated

                    chunk_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=sum(response_state.num_generated_tokens),
                        total_tokens=prompt_tokens + sum(response_state.num_generated_tokens),
                        tokens_per_second=max(response_state.tokens_per_second),
                        processing_time=max(response_state.time_spent_computing),
                    )
                    if new_tokens > 0:
                        self.metrics.total_tokens_generated += new_tokens
                        total_generated += new_tokens
                    choices = []
                    for idx in range(len(response_state.text)):
                        if idx not in accumulated_texts:
                            accumulated_texts[idx] = ""
                        accumulated_texts[idx] += response_state.text[idx]

                        if is_function_request and parser:
                            indicators = function_call_indicators.get(format_type, [])
                            might_be_function = any(ind in accumulated_texts[idx] for ind in indicators)

                            if might_be_function:
                                try:
                                    function_calls = parser.parse(accumulated_texts[idx])
                                    if function_calls:
                                        if idx not in partial_function_calls:
                                            partial_function_calls[idx] = []

                                        new_calls = function_calls[len(partial_function_calls[idx]) :]
                                        if new_calls:
                                            tool_calls = []
                                            for fc in new_calls:
                                                tool_call = ToolCall(
                                                    id=f"call_{idx}_{len(partial_function_calls[idx])}_{fc.name}",
                                                    type="function",
                                                    function=fc,
                                                )
                                                tool_calls.append(tool_call)
                                                partial_function_calls[idx].append(fc)

                                            choices.append(
                                                ChatCompletionStreamResponseChoice(
                                                    index=idx,
                                                    delta=DeltaMessage(tool_calls=tool_calls),
                                                    finish_reason=None,
                                                )
                                            )
                                            continue
                                except Exception:
                                    pass

                        if response_state.text[idx]:
                            choices.append(
                                ChatCompletionStreamResponseChoice(
                                    index=idx,
                                    delta=DeltaMessage(
                                        role="assistant" if response_state.num_generated_tokens[idx] == 1 else None,
                                        content=response_state.text[idx],
                                    ),
                                    finish_reason=None,
                                )
                            )

                    if choices:
                        chunk = ChatCompletionStreamResponse(model=request.model, choices=choices, usage=chunk_usage)
                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                if total_generated > 0:
                    generation_time = time.time() - start_time
                    tokens_per_second = total_generated / generation_time if generation_time > 0 else 0

                    final_choices = []
                    for idx in accumulated_texts:
                        if is_function_request and idx in partial_function_calls and partial_function_calls[idx]:
                            finish_reason = "tool_calls"
                        elif sum(response_state.num_generated_tokens) >= sampling_params.max_tokens:
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"

                        final_choices.append(
                            ChatCompletionStreamResponseChoice(
                                index=idx, delta=DeltaMessage(), finish_reason=finish_reason
                            )
                        )

                    usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=total_generated,
                        total_tokens=prompt_tokens + total_generated,
                        tokens_per_second=tokens_per_second,
                        processing_time=generation_time,
                        first_token_time=first_token_time,
                    )

                    final_chunk = ChatCompletionStreamResponse(
                        model=request.model,
                        choices=final_choices,
                        usage=usage,
                    )

                    yield f"data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception(f"Error during streaming: {e}")
                error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)
                yield f"data: {error_response.body.decode()}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id or "",
            },
        )

    async def list_tools(self) -> JSONResponse:
        """List available tools/functions for each model."""
        tools_by_model = {}

        for model_name, vsurge in self.vsurge_map.items():
            model_tools = getattr(vsurge, "available_tools", [])
            if not model_tools:
                model_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "example_function",
                            "description": "An example function for demonstration",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "param1": {"type": "string", "description": "First parameter"},
                                    "param2": {"type": "number", "description": "Second parameter"},
                                },
                                "required": ["param1"],
                            },
                        },
                    }
                ]

            tools_by_model[model_name] = {
                "tools": model_tools,
                "formats_supported": [
                    FunctionCallFormat.OPENAI.value,
                    FunctionCallFormat.HERMES.value,
                    FunctionCallFormat.GORILLA.value,
                    FunctionCallFormat.JSON_SCHEMA.value,
                ],
                "parallel_calls": True,
            }

        return JSONResponse(
            {
                "models": tools_by_model,
                "default_format": self.default_function_format.value,
            }
        )

    async def execute_tool(self, request: tp.Any) -> JSONResponse:
        """Execute a tool/function call (placeholder for integration)."""
        return create_error_response(
            HTTPStatus.NOT_IMPLEMENTED,
            "Tool execution endpoint is a placeholder. Implement based on your needs.",
        )

    def _create_sampling_params(self, request: ChatCompletionRequest | CompletionRequest) -> SamplingParams:
        """Create sampling parameters from request."""
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
            suppress_tokens=getattr(request, "suppress_tokens", None),
            stop=request.stop,
        )

    async def chat_completions(self, request: ChatCompletionRequest | ChatCompletionRequestWithTools) -> tp.Any:
        """Handle chat completion requests with function calling support."""
        request_id = getattr(request, "request_id", None)

        try:
            if not request.messages:
                raise HTTPException(400, "Messages cannot be empty")

            vsurge = self._get_vsurge(request.model)

            is_function_request = (
                self.enable_function_calling
                and isinstance(request, ChatCompletionRequestWithTools)
                and request.get_tools()
            )

            if is_function_request:
                content = await self._prepare_vsurge_input_with_tools_async(request, vsurge)
            else:
                content = await self._prepare_vsurge_input_async(request, vsurge)

            if request.stream:
                return await self._handle_streaming_with_tools_async(
                    request, vsurge, content, request_id, is_function_request
                )
            else:
                return await self._handle_completion_with_tools_async(
                    request, vsurge, content, request_id, is_function_request
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in chat completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def completions(self, request: CompletionRequest) -> tp.Any:
        """Handle completion requests."""
        request_id = getattr(request, "request_id", None)

        try:
            vsurge = self._get_vsurge(request.model)

            prompt = request.prompt
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""

            if not prompt:
                raise HTTPException(400, "Prompt cannot be empty")

            if request.stream:
                return await self._handle_streaming_response(request, vsurge, prompt, request_id)
            else:
                return await self._handle_completion_response(request, vsurge, prompt, request_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _prepare_chat_input(self, request: ChatCompletionRequest, vsurge: vSurge) -> str:
        """Prepare chat input for model."""
        loop = asyncio.get_event_loop()

        def _prepare():
            messages = [msg.model_dump() for msg in request.messages]
            processor = vsurge.processor

            if isinstance(processor, ProcessorMixin) and self.oai_like_processor:
                from easydel.trainers.prompt_utils import convert_to_openai_format

                messages = convert_to_openai_format(messages)

            if request.chat_template_kwargs is None:
                request.chat_template_kwargs = {}
            add_generation_prompt = request.chat_template_kwargs.pop("add_generation_prompt", True)
            return processor.apply_chat_template(
                tokenize=False,
                conversation=messages,
                add_generation_prompt=add_generation_prompt,
                **request.chat_template_kwargs,
            )

        return await loop.run_in_executor(self.thread_pool, _prepare)

    async def _handle_completion_response(
        self,
        request: ChatCompletionRequest | CompletionRequest,
        vsurge: vSurge,
        content: str,
        request_id: str | None = None,
    ) -> ChatCompletionResponse | CompletionResponse:
        """Generate non-streaming response."""
        start_time = time.time()

        try:
            prompt_tokens = await self._count_tokens_async(vsurge, content)
            sampling_params = self._create_sampling_params(request)
            response = await vsurge.generate(
                prompts=content,
                sampling_params=sampling_params,
                stream=False,
            )

            if not response:
                raise RuntimeError("Generation failed to produce output")

            result: ReturnSample = response[0]

            completion_tokens = sum(result.num_generated_tokens)
            self.metrics.total_tokens_generated += completion_tokens
            generation_time = time.time() - start_time
            tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
            if self.metrics.average_tokens_per_second == 0:
                self.metrics.average_tokens_per_second = tokens_per_second
            else:
                self.metrics.average_tokens_per_second = (
                    self.metrics.average_tokens_per_second * 0.9 + tokens_per_second * 0.1
                )

            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                tokens_per_second=tokens_per_second,
                processing_time=generation_time,
            )

            if isinstance(request, ChatCompletionRequest):
                return self._format_chat_response(request, result, usage)
            else:
                return self._format_completion_response(request, result, usage)

        except Exception as e:
            logger.exception(f"Error generating response: {e}")
            raise

    async def _handle_streaming_response(
        self,
        request: ChatCompletionRequest | CompletionRequest,
        vsurge: vSurge,
        content: str,
        request_id: str | None = None,
    ) -> StreamingResponse:
        """Generate streaming response."""

        async def generate_stream():
            start_time = time.time()
            prompt_tokens = await self._count_tokens_async(vsurge, content)
            sampling_params = self._create_sampling_params(request)

            try:
                total_tokens = 0
                first_token_time = None

                async for response_state in await vsurge.generate(
                    prompts=content,
                    sampling_params=sampling_params,
                    stream=True,
                ):
                    if not response_state:
                        continue

                    result: ReturnSample = response_state[0]

                    chunk_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=sum(response_state.num_generated_tokens),
                        total_tokens=prompt_tokens + sum(response_state.num_generated_tokens),
                        tokens_per_second=max(response_state.tokens_per_second),
                        processing_time=max(response_state.time_spent_computing),
                    )
                    if first_token_time is None and result.num_generated_tokens[0] > 0:
                        first_token_time = time.time() - start_time

                    current_tokens = sum(result.num_generated_tokens)
                    if current_tokens > total_tokens:
                        self.metrics.total_tokens_generated += current_tokens - total_tokens
                        total_tokens = current_tokens

                    if isinstance(request, ChatCompletionRequest):
                        chunk = self._format_chat_stream_chunk(request, result)
                    else:
                        chunk = self._format_completion_stream_chunk(request, result, chunk_usage)

                    yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                if total_tokens > 0:
                    generation_time = time.time() - start_time
                    tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

                    usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=total_tokens,
                        total_tokens=prompt_tokens + total_tokens,
                        tokens_per_second=tokens_per_second,
                        processing_time=generation_time,
                        first_token_time=first_token_time,
                    )

                    if isinstance(request, ChatCompletionRequest):
                        final_chunk = self._format_chat_final_chunk(request, usage)
                    else:
                        final_chunk = self._format_completion_final_chunk(request, usage)

                    yield f"data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception(f"Error during streaming: {e}")
                error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)
                yield f"data: {error_response.body.decode()}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id or "",
            },
        )

    async def _count_tokens_async(self, vsurge: vSurge, content: str) -> int:
        """Count tokens asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, vsurge.count_tokens, content)

    def _format_chat_response(
        self, request: ChatCompletionRequest, result: ReturnSample, usage: UsageInfo
    ) -> ChatCompletionResponse:
        """Format chat completion response."""
        choices = []
        for idx in range(len(result.text)):
            finish_reason = self._determine_finish_reason(
                result.num_generated_tokens[idx],
                request.max_tokens or float("inf"),
                result.accumulated_text[idx],
            )

            choices.append(
                ChatCompletionResponseChoice(
                    index=idx,
                    message=ChatMessage(
                        role="assistant",
                        content=result.accumulated_text[idx],
                    ),
                    finish_reason=finish_reason,
                )
            )

        return ChatCompletionResponse(
            model=request.model,
            choices=choices,
            usage=usage,
        )

    def _format_completion_response(
        self, request: CompletionRequest, result: ReturnSample, usage: UsageInfo
    ) -> CompletionResponse:
        """Format completion response."""
        choices = []
        for idx in range(len(result.text)):
            finish_reason = self._determine_finish_reason(
                result.num_generated_tokens[idx], request.max_tokens or float("inf"), result.accumulated_text[idx]
            )

            choices.append(
                CompletionResponseChoice(
                    index=idx,
                    text=result.accumulated_text[idx],
                    finish_reason=finish_reason,
                )
            )

        return CompletionResponse(
            model=request.model,
            choices=choices,
            usage=usage,
        )

    def _format_chat_stream_chunk(
        self, request: ChatCompletionRequest, result: ReturnSample
    ) -> ChatCompletionStreamResponse:
        """Format chat streaming chunk."""
        choices = []
        for idx in range(len(result.text)):
            choices.append(
                ChatCompletionStreamResponseChoice(
                    index=idx,
                    delta=DeltaMessage(
                        role="assistant" if result.num_generated_tokens[idx] == 1 else None,
                        content=result.text[idx] if result.text[idx] else None,
                    ),
                    finish_reason=None,
                )
            )

        return ChatCompletionStreamResponse(
            model=request.model,
            choices=choices,
        )

    def _format_completion_stream_chunk(
        self, request: CompletionRequest, result: ReturnSample, usage
    ) -> CompletionStreamResponse:
        """Format completion streaming chunk."""
        choices = []
        for idx in range(len(result.text)):
            choices.append(
                CompletionStreamResponseChoice(
                    index=idx,
                    text=result.text[idx],
                    finish_reason=None,
                )
            )

        return CompletionStreamResponse(
            model=request.model,
            choices=choices,
        )

    def _format_chat_final_chunk(self, request: ChatCompletionRequest, usage: UsageInfo) -> ChatCompletionStreamResponse:
        """Format final chat streaming chunk with usage info."""
        return ChatCompletionStreamResponse(
            model=request.model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    def _format_completion_final_chunk(self, request: CompletionRequest, usage: UsageInfo) -> CompletionStreamResponse:
        """Format final completion streaming chunk."""
        return CompletionStreamResponse(
            model=request.model,
            choices=[
                CompletionStreamResponseChoice(
                    index=0,
                    text="",
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    def _determine_finish_reason(self, tokens_generated: int, max_tokens: float, text: str) -> str:
        """Determine the finish reason for a generation."""
        if tokens_generated >= max_tokens:
            return "length"
        return "stop"

    async def health_check(self) -> JSONResponse:
        """Comprehensive health check."""
        health_status = {
            "status": self.status.value,
            "timestamp": time.time(),
            "uptime_seconds": self.metrics.uptime_seconds,
            "models": {
                name: {
                    "loaded": True,
                    "type": type(vsurge).__name__,
                }
                for name, vsurge in self.vsurge_map.items()
            },
            "active_requests": len(self._active_requests),
            "thread_pool": {
                "max_workers": self.thread_pool._max_workers,
                "active_threads": len(self.thread_pool._threads),
            },
        }

        status_code = 200 if self.status == ServerStatus.READY else 503
        return JSONResponse(health_status, status_code=status_code)

    async def list_models(self) -> JSONResponse:
        """List available models with function calling information."""
        models_data = []

        for model_id, vsurge in self.vsurge_map.items():
            model_info = {
                "id": model_id,
                "object": "model",
                "created": int(self.metrics.start_time),
                "owned_by": "easydel",
                "permission": [],
                "root": model_id,
                "parent": None,
                "metadata": {
                    "architecture": type(vsurge).__name__,
                    "supports_chat": hasattr(vsurge.processor, "apply_chat_template"),
                    "supports_function_calling": self.enable_function_calling,
                    "function_call_formats": [
                        FunctionCallFormat.OPENAI.value,
                        FunctionCallFormat.HERMES.value,
                        FunctionCallFormat.GORILLA.value,
                        FunctionCallFormat.JSON_SCHEMA.value,
                    ]
                    if self.enable_function_calling
                    else [],
                },
            }
            models_data.append(model_info)

        return JSONResponse(
            {
                "object": "list",
                "data": models_data,
                "total": len(models_data),
            }
        )

    async def get_model(self, model_id: str) -> JSONResponse:
        """Get detailed information about a specific model."""
        vsurge = self._get_vsurge(model_id)

        return JSONResponse(
            {
                "id": model_id,
                "object": "model",
                "created": int(self.metrics.start_time),
                "owned_by": "easydel",
                "permission": [],
                "root": model_id,
                "parent": None,
                "metadata": {
                    "architecture": type(vsurge).__name__,
                    "supports_chat": hasattr(vsurge.processor, "apply_chat_template"),
                },
            }
        )

    async def get_metrics(self) -> JSONResponse:
        """Get server performance metrics."""
        return JSONResponse(
            {
                "uptime_seconds": self.metrics.uptime_seconds,
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "total_tokens_generated": self.metrics.total_tokens_generated,
                "average_tokens_per_second": round(self.metrics.average_tokens_per_second, 2),
                "active_requests": len(self._active_requests),
                "models_loaded": len(self.vsurge_map),
                "status": self.status.value,
            }
        )

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
