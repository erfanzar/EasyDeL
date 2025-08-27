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
"""Implements an improved FastAPI server for serving vSurge models with OpenAI API compatibility."""

from __future__ import annotations

import asyncio
import time
import typing as tp
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
    CompletionStreamResponse,
    CompletionStreamResponseChoice,
    DeltaMessage,
    UsageInfo,
)
from ...sampling_params import SamplingParams
from ...tools.tool_calling_mixin import ToolCallingMixin
from ..utils import ReturnSample
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


class vSurgeAdapter(InferenceEngineAdapter):
    """Adapter for vSurge inference engine."""

    def __init__(self, vsurge_instance: vSurge):
        self.vsurge = vsurge_instance

    async def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> list[ReturnSample] | tp.AsyncGenerator[list[ReturnSample], None]:
        """Generate using vSurge."""
        return await self.vsurge.generate(prompts=prompts, sampling_params=sampling_params, stream=stream)

    def count_tokens(self, content: str) -> int:
        """Count tokens using vSurge tokenizer."""
        return self.vsurge.count_tokens(content)

    def get_model_info(self) -> dict[str, tp.Any]:
        """Get vSurge model information."""
        return {"name": self.vsurge.vsurge_name, "type": "vsurge", "architecture": type(self.vsurge).__name__}

    @property
    def model_name(self) -> str:
        return self.vsurge.vsurge_name

    @property
    def processor(self) -> tp.Any:
        return self.vsurge.processor


class vSurgeApiServer(BaseInferenceApiServer, ToolCallingMixin):
    """
    vSurge-specific API server implementation.

    This is now a clean implementation that follows the base interface.
    """

    def __init__(
        self,
        vsurge_map: dict[str, vSurge] | vSurge,
        oai_like_processor: bool = True,
        enable_function_calling: bool = True,
        tool_parser_name: str = "hermes",
        **kwargs,
    ) -> None:
        # Convert single instance to map
        if isinstance(vsurge_map, vSurge):
            vsurge_map = {vsurge_map.vsurge_name: vsurge_map}

        # Store both for backward compatibility
        self.vsurge_map = vsurge_map  # Keep original map
        self.adapters: dict[str, vSurgeAdapter] = {}

        # Build processor map for tool parser initialization
        model_processors = {}
        for name, vsurge in vsurge_map.items():
            if not isinstance(vsurge, vSurge):
                raise TypeError(f"Value for key '{name}' must be an instance of vSurge")
            self.adapters[name] = vSurgeAdapter(vsurge)
            model_processors[name] = vsurge.processor

        # Initialize tool parsers using mixin
        self.tool_parsers = self.initialize_tool_parsers(
            model_processors=model_processors,
            tool_parser_name=tool_parser_name,
            enable_function_calling=enable_function_calling,
        )

        self.oai_like_processor = oai_like_processor
        self.tool_parser_name = tool_parser_name

        super().__init__(
            server_name="EasyDeL vSurge API Server",
            server_description="High-performance vSurge inference server",
            enable_function_calling=enable_function_calling,
            **kwargs,
        )

    async def on_startup(self) -> None:
        """Custom startup logic for vSurge."""
        logger.info(f"Loaded {len(self.adapters)} vSurge models")
        for name in self.adapters:
            logger.info(f"  - {name}")

    def _get_adapter(self, model_name: str) -> vSurgeAdapter:
        """Get adapter by model name."""
        adapter = self.adapters.get(model_name)
        if adapter is None:
            available = list(self.adapters.keys())
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Available: {available}")
        return adapter

    def _count_tokens(self, content: str, model_name: str | None = None) -> int:
        """Count tokens for the given content."""
        if model_name:
            adapter = self._get_adapter(model_name)
            return adapter.count_tokens(content)
        adapter = next(iter(self.adapters.values()))
        return adapter.count_tokens(content)

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
            stop=request.stop,
        )

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
            logger.exception(f"Error applying chat template for model {vsurge.vsurge_name}: {e}")
            raise RuntimeError(f"Error tokenizing input: {e}") from e

    async def _prepare_vsurge_input_async(self, request, vsurge) -> dict:
        """Runs tokenization in the thread pool."""
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self._prepare_vsurge_input,
            request,
            vsurge,
        )

    async def _handle_completion(
        self,
        request: ChatCompletionRequest,
        vsurge: vSurge,
        content: str,
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
                if isinstance(response_text, list):
                    response_text = response_text[0]

                # Use mixin method for tool extraction
                message, finish_reason_extracted = self.extract_tool_calls_batch(
                    response_text=response_text,
                    request=request,
                    model_name=request.model,
                )

                # Override finish_reason if not a function call
                if finish_reason_extracted != "function_call":
                    finish_reason = self._determine_finish_reason(
                        result.num_generated_tokens[idx],
                        sampling_params.max_tokens,
                        response_text,
                    )
                else:
                    finish_reason = finish_reason_extracted

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

    async def _handle_streaming(self, request: ChatCompletionRequest, vsurge: vSurge, content: str) -> StreamingResponse:
        """Generate streaming response with function calling support."""

        async def generate_stream():
            start_time = time.time()
            prompt_tokens = vsurge.count_tokens(content)
            sampling_params = self._create_sampling_params(request)

            tool_parser = self.get_tool_parser_for_model(request.model)

            previous_text = ""

            try:
                total_generated = 0
                first_token_time = None
                accumulated_texts = {}

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

                        if tool_parser and idx == 0:  # Only process tool calls for first choice
                            current_text = accumulated_texts[idx]
                            delta_text = response_state.text[idx]

                            # Note: vSurge doesn't provide token_ids in the same way as eSurge
                            delta_message = self.extract_tool_calls_streaming(
                                model_name=request.model,
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                request=request,
                            )

                            previous_text = current_text

                            if delta_message:
                                if not delta_message.role:
                                    delta_message.role = (
                                        "assistant" if response_state.num_generated_tokens[idx] == 1 else None
                                    )
                            elif request.tools:
                                # Tool parser is active but returned None - it's buffering
                                # Don't send raw text that might contain tool markup
                                continue  # Skip this chunk entirely
                            else:
                                delta_message = DeltaMessage(
                                    role="assistant" if response_state.num_generated_tokens[idx] == 1 else None,
                                    content=response_state.text[idx],
                                )
                        else:
                            delta_message = DeltaMessage(
                                role="assistant" if response_state.num_generated_tokens[idx] == 1 else None,
                                content=response_state.text[idx],
                            )

                        if response_state.text[idx] or delta_message.tool_calls:
                            choices.append(
                                ChatCompletionStreamResponseChoice(
                                    index=idx,
                                    delta=delta_message,
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
                        if sum(response_state.num_generated_tokens) >= sampling_params.max_tokens:
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"

                        final_choices.append(
                            ChatCompletionStreamResponseChoice(
                                index=idx,
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

                    final_chunk = ChatCompletionStreamResponse(
                        model=request.model,
                        choices=final_choices,
                        usage=usage,
                    )

                    yield f"data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception(f"Error during streaming: {e}")
                error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
                yield f"data: {error_response.body.decode()}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def list_tools(self) -> JSONResponse:
        """List available tools/functions for each model.

        Returns example tool definitions and supported formats.
        This is a placeholder that can be extended with actual tools.
        """
        model_names = list(self.adapters.keys())
        tools_response = self.create_tools_response(model_names)
        return JSONResponse(tools_response)

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
            stop=request.stop,
        )

    async def chat_completions(self, request: ChatCompletionRequest) -> tp.Any:
        """Handle chat completion requests with function calling support."""
        request_id = getattr(request, "request_id", None)

        try:
            if not request.messages:
                raise HTTPException(400, "Messages cannot be empty")

            vsurge = self._get_adapter(request.model).vsurge

            content = await self._prepare_vsurge_input_async(request, vsurge)

            if request.stream:
                return await self._handle_streaming(request, vsurge, content)
            else:
                return await self._handle_completion(request, vsurge, content)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in chat completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def completions(self, request: CompletionRequest) -> tp.Any:
        """Handle completion requests."""
        try:
            vsurge = self._get_adapter(request.model).vsurge

            prompt = request.prompt
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""

            if not prompt:
                raise HTTPException(400, "Prompt cannot be empty")

            if request.stream:
                return await self._handle_streaming_response(request, vsurge, prompt)
            else:
                return await self._handle_completion_response(request, vsurge, prompt)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

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

                    response_state: ReturnSample = response_state[0]

                    chunk_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=sum(response_state.num_generated_tokens),
                        total_tokens=prompt_tokens + sum(response_state.num_generated_tokens),
                        tokens_per_second=max(response_state.tokens_per_second),
                        processing_time=max(response_state.time_spent_computing),
                    )
                    if first_token_time is None and response_state.num_generated_tokens[0] > 0:
                        first_token_time = time.time() - start_time

                    current_tokens = sum(response_state.num_generated_tokens)
                    if current_tokens > total_tokens:
                        self.metrics.total_tokens_generated += current_tokens - total_tokens
                        total_tokens = current_tokens

                    if isinstance(request, ChatCompletionRequest):
                        chunk = self._format_chat_stream_chunk(request, response_state)
                    else:
                        chunk = self._format_completion_stream_chunk(request, response_state, chunk_usage)

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
                error_response = create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
                yield f"data: {error_response.body.decode()}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def _count_tokens_async(self, vsurge: vSurge, content: str) -> int:
        """Count tokens asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, vsurge.count_tokens, content)

    def _format_chat_response(
        self,
        request: ChatCompletionRequest,
        result: ReturnSample,
        usage: UsageInfo,
    ) -> ChatCompletionResponse:
        """Format chat completion response."""
        choices = []
        for idx in range(len(result.text)):
            finish_reason = self._determine_finish_reason(
                result.num_generated_tokens[idx],
                request.max_tokens or float("inf"),
                result.accumulated_text[idx][0],
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
        self,
        request: CompletionRequest,
        result: ReturnSample,
        usage: UsageInfo,
    ) -> CompletionResponse:
        """Format completion response."""
        choices = []
        for idx in range(len(result.text)):
            finish_reason = self._determine_finish_reason(
                result.num_generated_tokens[idx],
                request.max_tokens or float("inf"),
                result.accumulated_text[idx],
            )
            text = result.accumulated_text[idx]
            if isinstance(text, list):
                text = text[0]
            choices.append(CompletionResponseChoice(index=idx, text=text, finish_reason=finish_reason))

        return CompletionResponse(model=request.model, choices=choices, usage=usage)

    def _format_chat_stream_chunk(
        self,
        request: ChatCompletionRequest,
        result: ReturnSample,
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
        model_health_info = {}
        for name, adapter in self.adapters.items():
            vsurge = adapter.vsurge
            device_memory_stats = None

            if hasattr(vsurge, "driver") and hasattr(vsurge.driver, "get_device_memory_stats"):
                try:
                    device_memory_stats = await vsurge.driver.get_device_memory_stats()
                except Exception as e:
                    logger.warning(f"Could not get device memory stats for model {name}: {e}")

            model_health_info[name] = {
                "loaded": True,
                "type": adapter.get_model_info()["type"],
                "architecture": adapter.get_model_info()["architecture"],
                "device_memory_stats": device_memory_stats or "N/A",
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
        """Get server performance metrics."""
        all_driver_metrics = {}

        for model_name, adapter in self.adapters.items():
            vsurge = adapter.vsurge
            if hasattr(vsurge, "driver") and hasattr(vsurge.driver, "get_metrics"):
                try:
                    all_driver_metrics[model_name] = vsurge.driver.get_metrics(aggregated=True)
                except Exception as e:
                    logger.warning(f"Failed to get driver metrics for {model_name}: {e}")

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
                "driver_metrics": all_driver_metrics,
            }
        )

    async def list_models(self) -> JSONResponse:
        """List available models."""
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
        """Get model details."""
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
