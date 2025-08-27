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
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus

from eformer.loggings import get_logger
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import ProcessorMixin

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

TIMEOUT_KEEP_ALIVE = 5.0
logger = get_logger("eSurgeApiServer")


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


class eSurgeApiServer(BaseInferenceApiServer, ToolCallingMixin):
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
    """

    def __init__(
        self,
        esurge_map: dict[str, eSurge] | eSurge,
        oai_like_processor: bool = True,
        enable_function_calling: bool = True,
        tool_parser_name: str = "hermes",
        **kwargs,
    ) -> None:
        """Initialize the eSurge API server.

        Args:
            esurge_map: Single eSurge instance or dict mapping model names to instances.
            oai_like_processor: Enable OpenAI-like processor compatibility for chat templates.
            enable_function_calling: Enable function/tool calling support.
            tool_parser_name: Name of the tool parser to use (e.g., "hermes", "qwen", etc.)
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
        self.status = ServerStatus.READY
        logger.info("eSurge API Server is ready")

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

    async def chat_completions(self, request: ChatCompletionRequest) -> tp.Any:
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
        self.metrics.total_requests += 1

        try:
            if not request.messages:
                raise HTTPException(400, "Messages cannot be empty")

            adapter = self._get_adapter(request.model)
            esurge = adapter.esurge

            content = await self._prepare_chat_input_async(request, esurge)

            if request.stream:
                return await self._handle_chat_streaming(request, esurge, content, request_id)
            else:
                return await self._handle_chat_completion(request, esurge, content, request_id)

        except HTTPException:
            self.metrics.failed_requests += 1
            raise
        except Exception as e:
            traceback.print_exc()
            self.metrics.failed_requests += 1
            logger.exception(f"Error in chat completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _handle_chat_completion(
        self,
        request: ChatCompletionRequest,
        esurge: eSurge,
        content: str,
        request_id: str,
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
        prompt_tokens = len(esurge.tokenizer(content)["input_ids"])

        sampling_params = self._create_sampling_params(request)
        outputs = esurge.generate(content, sampling_params, use_tqdm=False)

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

        self.metrics.successful_requests += 1

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

        async def generate_stream():
            prompt_tokens = len(esurge.tokenizer(content)["input_ids"])
            sampling_params = self._create_sampling_params(request)

            tool_parser = self.get_tool_parser_for_model(request.model)

            previous_text = ""
            previous_token_ids = []

            try:
                for output in esurge.stream(content, sampling_params):
                    current_completion_tokens = output.num_generated_tokens
                    current_tps = output.tokens_per_second
                    elapsed_time = output.processing_time

                    # Get delta message from tool parser if available
                    if tool_parser:
                        current_text = output.accumulated_text
                        delta_text = output.delta_text
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

                        # If tool parser returns a delta message, use it
                        if delta_message:
                            if not delta_message.role:
                                delta_message.role = "assistant"
                        elif request.tools:
                            # Tool parser is active but returned None - it's buffering
                            # Don't send raw text that might contain tool markup
                            continue  # Skip this chunk entirely
                        else:
                            # No special parsing needed for this chunk
                            delta_message = DeltaMessage(content=output.delta_text, role="assistant")
                    else:
                        # No tool parsing, regular streaming
                        delta_message = DeltaMessage(content=output.delta_text, role="assistant")

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
                    generation_time = output.processing_time
                    tokens_per_second = output.tokens_per_second
                    total_generated = output.num_generated_tokens

                usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=total_generated,
                    total_tokens=prompt_tokens + total_generated,
                    tokens_per_second=tokens_per_second,
                    processing_time=generation_time,
                    first_token_time=output.first_token_time,
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
                self.metrics.successful_requests += 1

            except Exception as e:
                self.metrics.failed_requests += 1
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

    async def completions(self, request: CompletionRequest) -> tp.Any:
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
        self.metrics.total_requests += 1

        try:
            adapter = self._get_adapter(request.model)
            esurge = adapter.esurge

            prompt = request.prompt
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""

            if not prompt:
                raise HTTPException(400, "Prompt cannot be empty")

            if request.stream:
                return await self._handle_completion_streaming(request, esurge, prompt, request_id)
            else:
                return await self._handle_completion_response(request, esurge, prompt, request_id)

        except HTTPException:
            self.metrics.failed_requests += 1
            raise
        except Exception as e:
            self.metrics.failed_requests += 1
            logger.exception(f"Error in completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _handle_completion_response(
        self,
        request: CompletionRequest,
        esurge: eSurge,
        prompt: str,
        request_id: str,
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
        prompt_tokens = len(esurge.tokenizer(prompt)["input_ids"])
        sampling_params = self._create_sampling_params(request)
        outputs = esurge.generate(prompt, sampling_params, use_tqdm=False)

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

        self.metrics.successful_requests += 1

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

        async def generate_stream():
            prompt_tokens = len(esurge.tokenizer(prompt)["input_ids"])
            sampling_params = self._create_sampling_params(request)
            try:
                for output in esurge.stream(prompt, sampling_params):
                    current_completion_tokens = output.num_generated_tokens
                    current_tps = output.tokens_per_second
                    elapsed_time = output.processing_time

                    chunk = ChatCompletionStreamResponse(
                        model=request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(content=output.delta_text, role="assistant"),
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

                    generation_time = output.processing_time
                    tokens_per_second = output.tokens_per_second
                    total_generated = output.num_generated_tokens

                usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=total_generated,
                    total_tokens=prompt_tokens + total_generated,
                    tokens_per_second=tokens_per_second,
                    processing_time=generation_time,
                    first_token_time=output.first_token_time,
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
                self.metrics.successful_requests += 1

            except Exception as e:
                self.metrics.failed_requests += 1
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

    async def health_check(self) -> JSONResponse:
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

    async def get_metrics(self) -> JSONResponse:
        """Get server performance metrics.

        Returns:
            JSONResponse with comprehensive server metrics including
            request counts, token statistics, throughput, and status.
        """
        self.metrics.uptime_seconds = time.time() - self.metrics.start_time

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
            }
        )

    async def list_models(self) -> JSONResponse:
        """List available models.

        OpenAI-compatible model listing endpoint.

        Returns:
            JSONResponse with list of available models and their metadata.
        """
        models_data = []
        for model_id, adapter in self.adapters.items():
            model_info = adapter.get_model_info()
            models_data.append(
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

        return JSONResponse(
            {
                "object": "list",
                "data": models_data,
                "total": len(models_data),
            }
        )

    async def get_model(self, model_id: str) -> JSONResponse:
        """Get model details.

        Args:
            model_id: Model identifier.

        Returns:
            JSONResponse with model metadata.

        Raises:
            HTTPException: If model not found.
        """
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

    async def list_tools(self) -> JSONResponse:
        """List available tools/functions for each model.

        Returns example tool definitions and supported formats.
        This is a placeholder that can be extended with actual tools.

        Returns:
            JSONResponse with tool definitions per model.
        """
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

    async def execute_tool(self, request: tp.Any) -> JSONResponse:
        """Execute a tool/function call.

        Placeholder endpoint for tool execution. Implement this method
        to integrate with actual tool execution systems.

        Args:
            request: Tool execution request.

        Returns:
            JSONResponse with NOT_IMPLEMENTED status.

        Note:
            This is a placeholder that should be implemented based on
            specific tool execution requirements.
        """
        return self.create_tool_execution_placeholder()
