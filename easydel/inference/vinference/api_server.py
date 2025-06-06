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
"""Enhanced FastAPI server for serving vInference models with OpenAI API compatibility."""

from __future__ import annotations

import asyncio
import json
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
from easydel.utils.helpers import get_logger

from ..openai_api_modules import (
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

if tp.TYPE_CHECKING:
    from ..vinference import vInference, vInferenceConfig
else:
    vInference = tp.Any
    vInferenceConfig = tp.Any

TIMEOUT_KEEP_ALIVE = 5.0
logger = get_logger("vInferenceApiServer")


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
    active_models: int = 0
    uptime_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)


@dataclass
class RequestContext:
    """Context for tracking request information."""

    request_id: str
    start_time: float
    model_name: str
    stream: bool = False


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


class vInferenceApiServer:
    """
    Enhanced FastAPI server for serving vInference instances.

    Features:
    - OpenAI API compatibility
    - Improved error handling and logging
    - Performance metrics tracking
    - Request validation and sanitization
    - Graceful shutdown handling
    - CORS support
    - Request ID tracking
    - Function calling support
    - Lazy model initialization
    """

    def __init__(
        self,
        inference_map: dict[str, vInference] | vInference | None = None,
        inference_init_call: tp.Callable[[], dict[str, vInference] | vInference] | None = None,
        max_workers: int = 10,
        allow_parallel_workload: bool = False,
        oai_like_processor: bool = True,
        enable_cors: bool = True,
        cors_origins: list[str] | None = None,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        request_timeout: float = 300.0,  # 5 minutes
        enable_function_calling: bool = True,
    ) -> None:
        """
        Initialize the vInference API Server.

        Args:
            inference_map: Dictionary of model names to vInference instances or single instance
            inference_init_call: Callable for lazy initialization of models
            max_workers: Maximum number of worker threads
            allow_parallel_workload: Allow parallel request processing
            oai_like_processor: Use OpenAI-like conversation format
            enable_cors: Enable CORS middleware
            cors_origins: Allowed CORS origins
            max_request_size: Maximum request size in bytes
            request_timeout: Request timeout in seconds
            enable_function_calling: Enable function calling support
        """
        from ..vinference import vInference

        # Initialize models
        if inference_init_call is not None:
            inference_map = inference_init_call()

        if inference_map is None:
            raise ValueError("Either `inference_map` or `inference_init_call` must be provided.")

        if isinstance(inference_map, vInference):
            inference_map = {inference_map.inference_name: inference_map}

        self.inference_map: dict[str, vInference] = {}
        self._validate_and_register_models(inference_map)

        # Server configuration
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vinference-worker")
        self.allow_parallel_workload = allow_parallel_workload
        self.oai_like_processor = oai_like_processor
        self.max_request_size = max_request_size
        self.request_timeout = request_timeout
        self.enable_function_calling = enable_function_calling

        # Server state
        self.status = ServerStatus.STARTING
        self.metrics = ServerMetrics(active_models=len(self.inference_map))
        self._active_requests: dict[str, RequestContext] = {}
        self._request_lock = asyncio.Lock()

        # Initialize FastAPI app
        self.app = FastAPI(
            title="EasyDeL vInference API Server",
            description="High-performance inference server with OpenAI API compatibility",
            version="2.0.0",
            lifespan=self._lifespan,
        )

        # Setup middleware
        if enable_cors:
            self._setup_cors(cors_origins)
        self._setup_middleware()

        # Register endpoints
        self._register_endpoints()

        logger.info(f"vInference API Server initialized with {len(self.inference_map)} models")

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Manage server lifecycle."""
        # Startup
        logger.info("Starting vInference API Server...")
        self.status = ServerStatus.READY
        yield
        # Shutdown
        logger.info("Shutting down vInference API Server...")
        self.status = ServerStatus.SHUTTING_DOWN
        await self._graceful_shutdown()

    def _validate_and_register_models(self, inference_map: dict[str, vInference]) -> None:
        """Validate and register vInference models."""
        from ..vinference import vInference

        for name, inference in inference_map.items():
            if not isinstance(inference, vInference):
                raise TypeError(
                    f"Value for key '{name}' must be an instance of vInference, got {type(inference).__name__}"
                )
            self.inference_map[name] = inference
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
        async def add_request_context(request: Request, call_next):
            """Add request context and tracking."""
            request_id = f"req_{int(time.time() * 1000000)}"
            request.state.request_id = request_id

            # Extract model name from request if possible
            model_name = "unknown"
            if request.url.path in ["/v1/chat/completions", "/v1/completions"]:
                try:
                    body = await request.body()
                    request._body = body  # Cache body for later use
                    data = json.loads(body)
                    model_name = data.get("model", "unknown")
                except Exception:
                    pass

            context = RequestContext(
                request_id=request_id,
                start_time=time.time(),
                model_name=model_name,
                stream=request.headers.get("accept") == "text/event-stream",
            )

            async with self._request_lock:
                self._active_requests[request_id] = context

            try:
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response
            finally:
                async with self._request_lock:
                    self._active_requests.pop(request_id, None)

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
            EndpointConfig(
                path="/v1/embeddings",
                handler=self.create_embeddings,
                methods=["POST"],
                tags=["Embeddings"],
                summary="Create embeddings",
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
        max_wait = 30  # seconds
        start = time.time()

        while self._active_requests and (time.time() - start) < max_wait:
            active_count = len(self._active_requests)
            logger.info(f"Waiting for {active_count} active requests to complete...")
            await asyncio.sleep(1)

        if self._active_requests:
            logger.warning(f"Force closing {len(self._active_requests)} active requests")

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        logger.info("Thread pool shut down")

    def _get_inference_model(self, model_name: str) -> vInference:
        """Get vInference instance by model name."""
        inference = self.inference_map.get(model_name)
        if inference is None:
            available = list(self.inference_map.keys())
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Available models: {available}")
        return inference

    def _create_sampling_params(self, request: ChatCompletionRequest | CompletionRequest) -> SamplingParams:
        """Create sampling parameters from request with validation."""
        # Validate and sanitize parameters
        max_tokens = min(int(request.max_tokens or 100), 4096)
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
            suppress_tokens=getattr(request, "suppress_tokens", None),
            stop=request.stop,
        )

    async def _prepare_tokenized_input_with_tools_async(
        self, request: ChatCompletionRequestWithTools, inference: vInference
    ) -> dict:
        """Prepare tokenized input with function/tool definitions."""
        loop = asyncio.get_event_loop()

        def _prepare():
            messages = [msg.model_dump() for msg in request.messages]
            processor = inference.processor_class

            if isinstance(processor, ProcessorMixin) and self.oai_like_processor:
                from easydel.trainers.prompt_utils import convert_to_openai_format

                messages = convert_to_openai_format(messages)

            tools = request.get_tools()
            if tools:
                tools_prompt = FunctionCallFormatter.format_tools_for_prompt(tools, request.function_call_format)

                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] += f"\n\n{tools_prompt}"
                else:
                    messages.insert(0, {"role": "system", "content": tools_prompt})

            if request.chat_template_kwargs is None:
                request.chat_template_kwargs = {}
            add_generation_prompt = request.chat_template_kwargs.pop("add_generation_prompt", True)

            return processor.apply_chat_template(
                conversation=messages,
                return_tensors="np",
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                tokenize=True,
                padding=True,
                **request.chat_template_kwargs,
            )

        return await loop.run_in_executor(self.thread_pool, _prepare)

    async def _handle_non_streaming_response_with_tools_async(
        self,
        request: ChatCompletionRequest | ChatCompletionRequestWithTools,
        inference: vInference,
        ids: dict,
        request_id: str | None = None,
        is_function_request: bool = False,
    ) -> ChatCompletionResponse:
        """Generate non-streaming response with function calling support."""
        start_time = time.time()

        try:
            prompt_tokens = ids["input_ids"].shape[-1]
            sampling_params = self._create_sampling_params(request)

            # Run generation
            response_state = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self._generate_response, inference, ids, sampling_params
            )

            if response_state is None:
                raise RuntimeError("Generation failed to produce output")

            # Decode generated text
            final_sequences = response_state.sequences
            generated_tokens = response_state.generated_tokens
            padded_length = response_state.padded_length

            final_responses = inference.tokenizer.batch_decode(
                final_sequences[..., padded_length:],
                skip_special_tokens=True,
            )

            # Update metrics
            self.metrics.total_tokens_generated += generated_tokens
            generation_time = time.time() - start_time
            tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0

            # Update average TPS
            if self.metrics.average_tokens_per_second == 0:
                self.metrics.average_tokens_per_second = tokens_per_second
            else:
                self.metrics.average_tokens_per_second = (
                    self.metrics.average_tokens_per_second * 0.9 + tokens_per_second * 0.1
                )

            # Create response choices
            choices = []
            for i, response_text in enumerate(final_responses):
                if is_function_request:
                    # Parse function calls from response
                    parser = FunctionCallParser(
                        format=getattr(request, "function_call_format", FunctionCallFormat.OPENAI), strict=False
                    )
                    function_calls = parser.parse(response_text)

                    if function_calls:
                        # Create message with tool calls
                        message = ChatMessageWithTools.from_function_calls(
                            function_calls,
                            content=None,  # No content when making function calls
                        )
                        finish_reason = "tool_calls"
                    else:
                        # Regular message if no function calls detected
                        message = ChatMessage(role="assistant", content=response_text)
                        finish_reason = self._determine_finish_reason(
                            generated_tokens, sampling_params.max_tokens, False
                        )
                else:
                    # Regular response without function calling
                    message = ChatMessage(role="assistant", content=response_text)
                    finish_reason = self._determine_finish_reason(generated_tokens, sampling_params.max_tokens, False)

                choices.append(
                    ChatCompletionResponseChoice(
                        index=i,
                        message=message,
                        finish_reason=finish_reason,
                    )
                )

            # Create usage info
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_tokens,
                total_tokens=prompt_tokens + generated_tokens,
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

    async def _handle_streaming_response_with_tools_async(
        self,
        request: ChatCompletionRequest | ChatCompletionRequestWithTools,
        inference: vInference,
        ids: dict,
        request_id: str | None = None,
        is_function_request: bool = False,
    ) -> StreamingResponse:
        """Generate streaming response with function calling support."""

        async def generate_stream():
            start_time = time.time()
            prompt_tokens = ids["input_ids"].shape[-1]
            sampling_params = self._create_sampling_params(request)

            # Function call parsing setup
            parser = None
            if is_function_request:
                parser = FunctionCallParser(
                    format=getattr(request, "function_call_format", FunctionCallFormat.OPENAI), strict=False
                )

            try:
                # Streaming state
                padded_length = None
                current_position = 0
                total_generated = 0
                first_token_time = None
                accumulated_text = [""] * ids["input_ids"].shape[0]
                partial_function_calls = [[] for _ in range(ids["input_ids"].shape[0])]
                function_call_started = [False] * ids["input_ids"].shape[0]

                async for response_state in self._async_generate(inference, ids, sampling_params):
                    padded_length = response_state.padded_length
                    current_position = padded_length

                    # Calculate new tokens
                    new_position = padded_length + response_state.generated_tokens
                    if new_position > current_position:
                        new_tokens = response_state.sequences[..., current_position:new_position]
                        decoded_chunks = await self._decode_tokens_async(inference, new_tokens)

                        # Track first token
                        if first_token_time is None and response_state.generated_tokens > 0:
                            first_token_time = time.time() - start_time

                        # Update metrics
                        tokens_generated = new_position - current_position
                        total_generated += tokens_generated
                        self.metrics.total_tokens_generated += tokens_generated

                        # Process each response
                        choices = []
                        for i, chunk_text in enumerate(decoded_chunks):
                            accumulated_text[i] += chunk_text

                            if is_function_request and parser:
                                # Check if this looks like the start of a function call
                                if not function_call_started[i]:
                                    # Look for function call indicators
                                    indicators = [
                                        '{"name"',
                                        "<function_call>",
                                        "<tool_call>",
                                        "Function call:",
                                        "```json",
                                        "<<<",
                                    ]
                                    if any(ind in accumulated_text[i] for ind in indicators):
                                        function_call_started[i] = True

                                if function_call_started[i]:
                                    # Try to parse partial function calls
                                    temp_calls = parser.parse(accumulated_text[i])
                                    if temp_calls and len(temp_calls) > len(partial_function_calls[i]):
                                        # New function call detected
                                        new_call = temp_calls[-1]
                                        partial_function_calls[i].append(new_call)

                                        # Stream the function call
                                        tool_call = ToolCall(
                                            id=f"call_{i}_{len(partial_function_calls[i])}_{new_call.name}",
                                            type="function",
                                            function=new_call,
                                        )

                                        choices.append(
                                            ChatCompletionStreamResponseChoice(
                                                index=i,
                                                delta=DeltaMessage(tool_calls=[tool_call]),
                                                finish_reason=None,
                                            )
                                        )
                                        continue

                            # Regular text streaming
                            choices.append(
                                ChatCompletionStreamResponseChoice(
                                    index=i,
                                    delta=DeltaMessage(
                                        role="assistant" if current_position == padded_length else None,
                                        content=chunk_text if chunk_text and not function_call_started[i] else None,
                                    ),
                                    finish_reason=None,
                                )
                            )

                        if choices:
                            chunk = ChatCompletionStreamResponse(
                                model=request.model,
                                choices=choices,
                                usage=UsageInfo(
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=response_state.generated_tokens,
                                    total_tokens=prompt_tokens + response_state.generated_tokens,
                                    tokens_per_second=response_state.tokens_per_second,
                                    processing_time=response_state._time_spent_computing,
                                    first_token_time=first_token_time,
                                ),
                            )
                            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                        current_position = new_position

                # Send final chunk
                if total_generated > 0:
                    generation_time = time.time() - start_time
                    tokens_per_second = total_generated / generation_time if generation_time > 0 else 0

                    final_choices = []
                    for i, full_text in enumerate(accumulated_text):  # noqa
                        if is_function_request and partial_function_calls[i]:
                            finish_reason = "tool_calls"
                        elif total_generated >= sampling_params.max_tokens:
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"

                        final_choices.append(
                            ChatCompletionStreamResponseChoice(
                                index=i,
                                delta=DeltaMessage(),
                                finish_reason=finish_reason,
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

                    final_chunk = ChatCompletionStreamResponse(model=request.model, choices=final_choices, usage=usage)

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

    def _determine_finish_reason(self, tokens_generated: int, max_tokens: int, has_function_call: bool) -> str:
        """Determine the finish reason for generation."""
        if has_function_call:
            return "tool_calls"  # Updated to match OpenAI's format
        elif tokens_generated >= max_tokens:
            return "length"
        else:
            return "stop"

    # Add endpoint for handling tool/function results
    async def submit_tool_outputs(self, request: tp.Any) -> tp.Any:
        """Handle tool output submissions for multi-turn function calling."""
        # This would handle the results of function calls and continue the conversation
        # Implementation depends on your specific requirements
        return create_error_response(HTTPStatus.NOT_IMPLEMENTED, "Tool output submission not yet implemented")

    async def chat_completions(self, request: ChatCompletionRequest | ChatCompletionRequestWithTools) -> tp.Any:
        """Handle chat completion requests with function calling support."""
        request_id = getattr(request, "request_id", None)

        try:
            # Validate request
            if not request.messages:
                raise HTTPException(400, "Messages cannot be empty")

            inference = self._get_inference_model(request.model)

            # Check if this is a function calling request
            is_function_request = isinstance(request, ChatCompletionRequestWithTools) and request.get_tools()

            # Prepare input with function definitions if needed
            if is_function_request:
                ids = await self._prepare_tokenized_input_with_tools_async(request, inference)
            else:
                ids = await self._prepare_tokenized_input_async(request, inference)

            # Generate response
            if request.stream:
                return await self._handle_streaming_response_with_tools_async(
                    request, inference, ids, request_id, is_function_request
                )
            else:
                return await self._handle_non_streaming_response_with_tools_async(
                    request, inference, ids, request_id, is_function_request
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in chat completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def completions(self, request: CompletionRequest) -> tp.Any:
        """Handle completion requests with enhanced validation."""
        request_id = getattr(request, "request_id", None)

        try:
            inference = self._get_inference_model(request.model)

            # Process prompt
            prompt = request.prompt
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""

            if not prompt:
                raise HTTPException(400, "Prompt cannot be empty")

            # Tokenize prompt
            inputs = await self._tokenize_prompt_async(inference, prompt)

            # Generate response
            if request.stream:
                return await self._handle_completion_streaming_async(request, inference, inputs, request_id)
            else:
                return await self._handle_completion_response_async(request, inference, inputs, request_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _tokenize_prompt_async(self, inference: vInference, prompt: str) -> dict:
        """Tokenize a prompt asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: inference.tokenizer(prompt, return_tensors="np", padding=True),
        )

    def _prepare_tokenized_input(
        self,
        request: ChatCompletionRequest,
        inference: vInference,
    ) -> dict:
        """Prepare and tokenize chat input."""
        messages = [msg.model_dump() for msg in request.messages]
        processor = inference.processor_class

        if isinstance(processor, ProcessorMixin) and self.oai_like_processor:
            from easydel.trainers.prompt_utils import convert_to_openai_format

            messages = convert_to_openai_format(messages)

        try:
            if request.chat_template_kwargs is None:
                request.chat_template_kwargs = {}
            add_generation_prompt = request.chat_template_kwargs.pop("add_generation_prompt", True)
            return processor.apply_chat_template(
                conversation=messages,
                return_tensors="np",
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                tokenize=True,
                padding=True,
                **request.chat_template_kwargs,
            )
        except Exception as e:
            logger.exception(f"Error applying chat template: {e}")
            raise RuntimeError(f"Error tokenizing input: {e}") from e

    async def _prepare_tokenized_input_async(self, request: ChatCompletionRequest, inference: vInference) -> dict:
        """Prepare tokenized input asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, self._prepare_tokenized_input, request, inference)

    def _extract_function_call(self, text: str) -> dict | None:
        """Extract function call from generated text."""
        if not self.enable_function_calling:
            return None

        try:
            # Look for JSON-like function calls in the response
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                possible_json = text[start:end]

                parsed = json.loads(possible_json)
                if "name" in parsed and ("arguments" in parsed or "params" in parsed):
                    return {
                        "name": parsed.get("name"),
                        "arguments": json.dumps(parsed.get("arguments", parsed.get("params", {}))),
                    }
        except (json.JSONDecodeError, Exception):
            pass

        return None

    async def _handle_non_streaming_response_async(
        self, request: ChatCompletionRequest, inference: vInference, ids: dict, request_id: str | None = None
    ) -> ChatCompletionResponse:
        """Generate non-streaming chat response with enhanced metrics."""
        start_time = time.time()

        try:
            prompt_tokens = ids["input_ids"].shape[-1]
            sampling_params = self._create_sampling_params(request)

            # Run generation in thread pool
            loop = asyncio.get_event_loop()
            response_state = await loop.run_in_executor(
                self.thread_pool, self._generate_response, inference, ids, sampling_params
            )

            if response_state is None:
                raise RuntimeError("Generation failed to produce output")

            # Extract results
            final_sequences = response_state.sequences
            generated_tokens = response_state.generated_tokens
            padded_length = response_state.padded_length

            # Decode generated text
            final_responses = inference.tokenizer.batch_decode(
                final_sequences[..., padded_length:],
                skip_special_tokens=True,
            )

            # Update metrics
            self.metrics.total_tokens_generated += generated_tokens
            generation_time = time.time() - start_time
            tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0

            # Update average TPS with exponential moving average
            if self.metrics.average_tokens_per_second == 0:
                self.metrics.average_tokens_per_second = tokens_per_second
            else:
                self.metrics.average_tokens_per_second = (
                    self.metrics.average_tokens_per_second * 0.9 + tokens_per_second * 0.1
                )

            # Create response choices
            choices = []
            for i, response_text in enumerate(final_responses):
                # Check for function calls
                function_call = None
                content = response_text

                if hasattr(request, "functions") and request.functions:
                    function_call = self._extract_function_call(response_text)
                    if function_call:
                        content = None

                finish_reason = self._determine_finish_reason(
                    generated_tokens, sampling_params.max_tokens, function_call is not None
                )

                choices.append(
                    ChatCompletionResponseChoice(
                        index=i,
                        message=ChatMessage(
                            role="assistant",
                            content=content,
                            function_call=function_call,
                        ),
                        finish_reason=finish_reason,
                    )
                )

            # Create usage info
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_tokens,
                total_tokens=prompt_tokens + generated_tokens,
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

    def _generate_response(self, inference: vInference, ids: dict, sampling_params: SamplingParams) -> tp.Any:
        """Generate response synchronously."""
        response_state = None
        for response_state in inference.generate(**ids, sampling_params=sampling_params):  # noqa
            pass  # Iterate to get final state
        return response_state

    async def _handle_streaming_response_async(
        self, request: ChatCompletionRequest, inference: vInference, ids: dict, request_id: str | None = None
    ) -> StreamingResponse:
        """Generate streaming chat response with improved chunk handling."""

        async def generate_stream():
            start_time = time.time()
            prompt_tokens = ids["input_ids"].shape[-1]
            sampling_params = self._create_sampling_params(request)

            try:
                # Setup streaming state
                padded_length = None
                current_position = 0
                total_generated = 0
                first_token_time = None
                accumulated_text = [""] * ids["input_ids"].shape[0]

                # Create async generator from sync generator
                async for response_state in self._async_generate(inference, ids, sampling_params):
                    padded_length = response_state.padded_length
                    current_position = padded_length

                    # Calculate new tokens in this chunk
                    new_position = padded_length + response_state.generated_tokens
                    if new_position > current_position:
                        # Decode only new tokens
                        new_tokens = response_state.sequences[..., current_position:new_position]
                        decoded_chunks = await self._decode_tokens_async(inference, new_tokens)

                        # Track first token time
                        if first_token_time is None and response_state.generated_tokens > 0:
                            first_token_time = time.time() - start_time

                        # Update metrics
                        tokens_generated = new_position - current_position
                        total_generated += tokens_generated
                        self.metrics.total_tokens_generated += tokens_generated

                        # Create streaming choices
                        choices = []
                        for i, chunk_text in enumerate(decoded_chunks):
                            accumulated_text[i] += chunk_text

                            choices.append(
                                ChatCompletionStreamResponseChoice(
                                    index=i,
                                    delta=DeltaMessage(
                                        role="assistant" if current_position == padded_length else None,
                                        content=chunk_text if chunk_text else None,
                                    ),
                                    finish_reason=None,
                                )
                            )

                        # Create and yield chunk
                        chunk = ChatCompletionStreamResponse(
                            model=request.model,
                            choices=choices,
                        )

                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                        current_position = new_position

                # Send final chunk with usage info
                if total_generated > 0:
                    generation_time = time.time() - start_time
                    tokens_per_second = total_generated / generation_time if generation_time > 0 else 0

                    # Check for function calls in accumulated text
                    final_choices = []
                    for i, full_text in enumerate(accumulated_text):
                        function_call = None
                        finish_reason = "stop"

                        if hasattr(request, "functions") and request.functions:
                            function_call = self._extract_function_call(full_text)
                            if function_call:
                                finish_reason = "function_call"
                        elif total_generated >= sampling_params.max_tokens:
                            finish_reason = "length"

                        final_choices.append(
                            ChatCompletionStreamResponseChoice(
                                index=i,
                                delta=DeltaMessage(function_call=function_call),
                                finish_reason=finish_reason,
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

    async def _async_generate(self, inference: vInference, ids: dict, sampling_params: SamplingParams):
        """Convert synchronous generator to async generator."""
        loop = asyncio.get_event_loop()

        def _sync_gen():
            yield from inference.generate(**ids, sampling_params=sampling_params)

        gen = await loop.run_in_executor(self.thread_pool, _sync_gen)

        for item in gen:
            yield item
            if self.allow_parallel_workload:
                await asyncio.sleep(0)

    async def _decode_tokens_async(self, inference: vInference, tokens: tp.Any) -> list[str]:
        """Decode tokens asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: inference.tokenizer.batch_decode(tokens, skip_special_tokens=True),
        )

    async def _handle_completion_response_async(
        self, request: CompletionRequest, inference: vInference, inputs: dict, request_id: str | None = None
    ) -> CompletionResponse:
        """Generate non-streaming completion response."""
        start_time = time.time()

        try:
            prompt_tokens = inputs["input_ids"].shape[-1]
            sampling_params = self._create_sampling_params(request)

            # Generate response
            response_state = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self._generate_response, inference, inputs, sampling_params
            )

            if response_state is None:
                raise RuntimeError("Generation failed to produce output")

            # Decode completions
            completions = inference.tokenizer.batch_decode(
                response_state.sequences[..., response_state.padded_length :],
                skip_special_tokens=True,
            )

            # Update metrics
            generated_tokens = response_state.generated_tokens
            self.metrics.total_tokens_generated += generated_tokens
            generation_time = time.time() - start_time

            # Create response
            choices = [
                CompletionResponseChoice(
                    text=completion,
                    index=i,
                    finish_reason=self._determine_finish_reason(generated_tokens, sampling_params.max_tokens, False),
                )
                for i, completion in enumerate(completions)
            ]

            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_tokens,
                total_tokens=prompt_tokens + generated_tokens,
                tokens_per_second=generated_tokens / generation_time if generation_time > 0 else 0,
                processing_time=generation_time,
            )

            return CompletionResponse(
                model=request.model,
                choices=choices,
                usage=usage,
            )

        except Exception as e:
            logger.exception(f"Error generating completion: {e}")
            raise

    async def _handle_completion_streaming_async(
        self,
        request: CompletionRequest,
        inference: vInference,
        inputs: dict,
        request_id: str | None = None,
    ) -> StreamingResponse:
        """Generate streaming completion response."""

        async def generate_stream():
            start_time = time.time()
            prompt_tokens = inputs["input_ids"].shape[-1]
            sampling_params = self._create_sampling_params(request)

            try:
                padded_length = None
                current_position = 0
                total_generated = 0
                first_token_time = None
                generation_time = 0
                tokens_per_second = 0
                async for response_state in self._async_generate(inference, inputs, sampling_params):
                    padded_length = response_state.padded_length
                    current_position = padded_length

                    generation_time = response_state._time_spent_computing
                    tokens_per_second = response_state.tokens_per_second
                    new_position = padded_length + response_state.generated_tokens
                    if new_position > current_position:
                        new_tokens = response_state.sequences[..., current_position:new_position]
                        decoded_chunks = await self._decode_tokens_async(inference, new_tokens)

                        if first_token_time is None and response_state.generated_tokens > 0:
                            first_token_time = time.time() - start_time

                        tokens_generated = new_position - current_position
                        total_generated += tokens_generated
                        self.metrics.total_tokens_generated += tokens_generated

                        choices = [
                            CompletionStreamResponseChoice(
                                text=chunk_text,
                                index=i,
                                finish_reason=None,
                            )
                            for i, chunk_text in enumerate(decoded_chunks)
                        ]

                        # Yield chunk
                        chunk = CompletionStreamResponse(
                            model=request.model,
                            choices=choices,
                            usage=UsageInfo(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=total_generated,
                                total_tokens=prompt_tokens + total_generated,
                                tokens_per_second=tokens_per_second,
                                processing_time=generation_time,
                                first_token_time=first_token_time,
                            ),
                        )

                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                        current_position = new_position

                # Send final chunk
                if total_generated > 0:
                    finish_reason = "length" if total_generated >= sampling_params.max_tokens else "stop"

                    final_choices = [
                        CompletionStreamResponseChoice(
                            text="",
                            index=i,
                            finish_reason=finish_reason,
                        )
                        for i in range(inputs["input_ids"].shape[0])
                    ]

                    usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=total_generated,
                        total_tokens=prompt_tokens + total_generated,
                        tokens_per_second=tokens_per_second,
                        processing_time=generation_time,
                        first_token_time=first_token_time,
                    )

                    final_chunk = CompletionStreamResponse(model=request.model, choices=final_choices, usage=usage)

                    yield f"data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception(f"Error during streaming completion: {e}")
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

    def _determine_finish_reason(self, tokens_generated: int, max_tokens: int, has_function_call: bool) -> str:
        """Determine the finish reason for generation."""
        if has_function_call:
            return "tool_calls"
        elif tokens_generated >= max_tokens:
            return "length"
        else:
            return "stop"

    async def create_embeddings(self, request: tp.Any) -> JSONResponse:
        """Create embeddings endpoint (placeholder for future implementation)."""
        return create_error_response(HTTPStatus.NOT_IMPLEMENTED, "Embeddings endpoint not yet implemented")

    async def health_check(self) -> JSONResponse:
        """Comprehensive health check endpoint."""
        health_status = {
            "status": self.status.value,
            "timestamp": time.time(),
            "uptime_seconds": self.metrics.uptime_seconds,
            "models": {},
            "system": {
                "active_requests": len(self._active_requests),
                "thread_pool": {
                    "max_workers": self.thread_pool._max_workers,
                    "active_threads": len(self.thread_pool._threads),
                },
            },
        }

        # Check each model's health
        for name, inference in self.inference_map.items():
            try:
                # Basic health check - verify model is accessible
                health_status["models"][name] = {
                    "loaded": True,
                    "type": type(inference).__name__,
                    "processor": type(inference.processor_class).__name__,
                    "supports_streaming": hasattr(inference.generation_config, "streaming_chunks"),
                }
            except Exception as e:
                health_status["models"][name] = {"loaded": False, "error": str(e)}
                self.status = ServerStatus.ERROR

        # Determine overall health
        all_models_healthy = all(model.get("loaded", False) for model in health_status["models"].values())

        if self.status == ServerStatus.SHUTTING_DOWN:
            status_code = 503
        elif all_models_healthy and self.status != ServerStatus.ERROR:
            status_code = 200
            self.status = ServerStatus.READY
        else:
            status_code = 503

        return JSONResponse(health_status, status_code=status_code)

    async def list_models(self) -> JSONResponse:
        """List available models with detailed information."""
        models_data = []

        for model_id, inference in self.inference_map.items():
            try:
                model_info = {
                    "id": model_id,
                    "object": "model",
                    "created": int(self.metrics.start_time),
                    "owned_by": "easydel",
                    "permission": [],
                    "root": model_id,
                    "parent": None,
                    "metadata": {
                        "architecture": type(inference).__name__,
                        "supports_chat": hasattr(inference.processor_class, "apply_chat_template"),
                        "supports_streaming": hasattr(inference.generation_config, "streaming_chunks"),
                        "max_tokens": getattr(inference.generation_config, "max_new_tokens", None),
                        "supports_function_calling": self.enable_function_calling,
                    },
                }
                models_data.append(model_info)
            except Exception as e:
                logger.error(f"Error getting info for model {model_id}: {e}")

        return JSONResponse(
            {
                "object": "list",
                "data": models_data,
                "total": len(models_data),
            }
        )

    async def get_model(self, model_id: str) -> JSONResponse:
        """Get detailed information about a specific model."""
        try:
            inference = self._get_inference_model(model_id)

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
                        "architecture": type(inference).__name__,
                        "supports_chat": hasattr(inference.processor_class, "apply_chat_template"),
                        "supports_streaming": hasattr(inference.generation_config, "streaming_chunks"),
                        "max_tokens": getattr(inference.generation_config, "max_new_tokens", None),
                        "supports_function_calling": self.enable_function_calling,
                    },
                }
            )
        except HTTPException:
            raise
        except Exception as e:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error retrieving model info: {e!s}")

    async def get_metrics(self) -> JSONResponse:
        """Get detailed server performance metrics."""
        # Calculate request statistics
        active_requests_by_model = {}
        for req in self._active_requests.values():
            model = req.model_name
            if model not in active_requests_by_model:
                active_requests_by_model[model] = 0
            active_requests_by_model[model] += 1

        return JSONResponse(
            {
                "uptime_seconds": round(self.metrics.uptime_seconds, 2),
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": round(self.metrics.successful_requests / max(self.metrics.total_requests, 1) * 100, 2),
                "total_tokens_generated": self.metrics.total_tokens_generated,
                "average_tokens_per_second": round(self.metrics.average_tokens_per_second, 2),
                "active_requests": {
                    "total": len(self._active_requests),
                    "by_model": active_requests_by_model,
                },
                "models": {
                    "loaded": len(self.inference_map),
                    "names": list(self.inference_map.keys()),
                },
                "server": {
                    "status": self.status.value,
                    "version": "2.0.0",
                    "features": {
                        "streaming": True,
                        "function_calling": self.enable_function_calling,
                        "parallel_workload": self.allow_parallel_workload,
                        "openai_compatibility": self.oai_like_processor,
                    },
                },
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
        access_log: bool = True,
    ) -> None:
        """
        Start the server with enhanced configuration.

        Args:
            host: Host address to bind to
            port: Port to listen on
            workers: Number of worker processes (ignored if reload=True)
            log_level: Logging level
            ssl_keyfile: Path to SSL key file
            ssl_certfile: Path to SSL certificate file
            reload: Enable auto-reload for development
            access_log: Enable access logging
        """
        uvicorn_config = {
            "app": self.app,
            "host": host,
            "port": port,
            "workers": workers if not reload else 1,
            "log_level": log_level,
            "timeout_keep_alive": TIMEOUT_KEEP_ALIVE,
            "reload": reload,
            "access_log": access_log,
            "server_header": False,  # Security: hide server header
            "date_header": True,
            "limit_concurrency": 1000,  # Prevent overload
            "limit_max_requests": 10000,  # Restart workers periodically
        }

        # Configure SSL if provided
        if ssl_keyfile and ssl_certfile:
            uvicorn_config.update(
                {
                    "ssl_keyfile": ssl_keyfile,
                    "ssl_certfile": ssl_certfile,
                    "ssl_version": 3,  # TLS 1.2+
                }
            )
            logger.info(f"Starting HTTPS server on https://{host}:{port}")
        else:
            logger.info(f"Starting HTTP server on http://{host}:{port}")

        # Use uvloop for better performance if available
        try:
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Using uvloop for enhanced performance")
        except ImportError:
            logger.info("uvloop not available, using default event loop")

        # Log server configuration
        logger.info("Server configuration:")
        logger.info(f"  - Models: {list(self.inference_map.keys())}")
        logger.info(f"  - Workers: {uvicorn_config['workers']}")
        logger.info(f"  - Max thread pool workers: {self.thread_pool._max_workers}")
        logger.info(f"  - Features: streaming={True}, function_calling={self.enable_function_calling}")

        # Start server
        uvicorn.run(**uvicorn_config)

    fire = run
