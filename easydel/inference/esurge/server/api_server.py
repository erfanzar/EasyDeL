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
import typing as tp
import uuid
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import ProcessorMixin

from easydel.utils.helpers import get_logger

from ...inference_engine_interface import BaseInferenceApiServer, InferenceEngineAdapter
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
    UsageInfo,
)
from ...sampling_params import SamplingParams
from ..esurge_engine import RequestOutput, eSurge

TIMEOUT_KEEP_ALIVE = 5.0
logger = get_logger("eSurgeApiServer")


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


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: dict[str, str]
    request_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


def create_error_response(status_code: HTTPStatus, message: str, request_id: str | None = None) -> JSONResponse:
    """Creates a standardized JSON error response."""
    error_response = ErrorResponse(error={"message": message, "type": status_code.name}, request_id=request_id)
    return JSONResponse(content=error_response.model_dump(), status_code=status_code.value)


class eSurgeAdapter(InferenceEngineAdapter):
    """Adapter for eSurge inference engine."""

    def __init__(self, esurge_instance: eSurge, model_name: str):
        self.esurge = esurge_instance
        self._model_name = model_name

    async def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> list[RequestOutput] | tp.AsyncGenerator[RequestOutput, None]:
        """Generate using eSurge."""
        if stream:
            async def stream_generator():
                if isinstance(prompts, str):
                    prompt_list = [prompts]
                else:
                    prompt_list = prompts

                for prompt in prompt_list:
                    for output in self.esurge.stream(prompt, sampling_params):
                        yield output

            return stream_generator()
        else:
            return self.esurge.generate(prompts, sampling_params)

    def count_tokens(self, content: str) -> int:
        """Count tokens using eSurge tokenizer."""
        return len(self.esurge.tokenizer(content)["input_ids"])

    def get_model_info(self) -> dict[str, tp.Any]:
        """Get eSurge model information."""
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


class eSurgeApiServer(BaseInferenceApiServer):
    """
    eSurge-specific API server implementation with OpenAI compatibility.
    """

    def __init__(
        self,
        esurge_map: dict[str, eSurge] | eSurge,
        oai_like_processor: bool = True,
        enable_function_calling: bool = True,
        **kwargs,
    ) -> None:
        if isinstance(esurge_map, eSurge):
            model_name = esurge_map.esurge_name
            esurge_map = {model_name: esurge_map}

        self.esurge_map = esurge_map
        self.adapters: dict[str, eSurgeAdapter] = {}

        for name, esurge in esurge_map.items():
            if not isinstance(esurge, eSurge):
                raise TypeError(f"Value for key '{name}' must be an instance of eSurge")
            self.adapters[name] = eSurgeAdapter(esurge, name)

        self.oai_like_processor = oai_like_processor
        self.metrics = ServerMetrics()
        self.status = ServerStatus.STARTING
        self._active_requests: dict[str, dict] = {}

        if "default_function_format" not in kwargs:
            kwargs["default_function_format"] = FunctionCallFormat.QWEN

        super().__init__(
            server_name="EasyDeL eSurge API Server",
            server_description="High-performance eSurge inference server with OpenAI compatibility",
            enable_function_calling=enable_function_calling,
            **kwargs,
        )

    async def on_startup(self) -> None:
        """Custom startup logic for eSurge."""
        logger.info(f"Loaded {len(self.adapters)} eSurge models")
        for name in self.adapters:
            logger.info(f"  - {name}")
        self.status = ServerStatus.READY
        logger.info("eSurge API Server is ready")

    def _get_adapter(self, model_name: str) -> eSurgeAdapter:
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

    def _prepare_chat_input(
        self,
        request: ChatCompletionRequest,
        esurge: eSurge,
    ) -> str:
        """Prepare chat input for model."""
        conversation = request.model_dump(exclude_unset=True)["messages"]
        processor = esurge.tokenizer

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
            logger.exception(f"Error applying chat template: {e}")
            raise RuntimeError(f"Error tokenizing input: {e}") from e

    async def _prepare_chat_input_async(self, request: ChatCompletionRequest, esurge: eSurge) -> str:
        """Prepare chat input asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._prepare_chat_input,
            request,
            esurge,
        )

    def _prepare_chat_input_with_tools(
        self,
        request: ChatCompletionRequestWithTools,
        esurge: eSurge,
    ) -> str:
        """Prepare input with function/tool definitions."""
        messages = [msg.model_dump() for msg in request.messages]
        processor = esurge.tokenizer

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

    async def _prepare_chat_input_with_tools_async(
        self,
        request: ChatCompletionRequestWithTools,
        esurge: eSurge,
    ) -> str:
        """Prepare input with tools asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._prepare_chat_input_with_tools,
            request,
            esurge,
        )

    async def chat_completions(self, request: ChatCompletionRequest | ChatCompletionRequestWithTools) -> tp.Any:
        """Handle chat completion requests."""
        request_id = str(uuid.uuid4())
        self.metrics.total_requests += 1

        try:
            if not request.messages:
                raise HTTPException(400, "Messages cannot be empty")

            adapter = self._get_adapter(request.model)
            esurge = adapter.esurge

            is_function_request = (
                self.enable_function_calling
                and isinstance(request, ChatCompletionRequestWithTools)
                and request.get_tools()
            )

            if is_function_request:
                content = await self._prepare_chat_input_with_tools_async(request, esurge)
            else:
                content = await self._prepare_chat_input_async(request, esurge)

            if request.stream:
                return await self._handle_chat_streaming(request, esurge, content, request_id, is_function_request)
            else:
                return await self._handle_chat_completion(request, esurge, content, request_id, is_function_request)

        except HTTPException:
            self.metrics.failed_requests += 1
            raise
        except Exception as e:
            self.metrics.failed_requests += 1
            logger.exception(f"Error in chat completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _handle_chat_completion(
        self,
        request: ChatCompletionRequest,
        esurge: eSurge,
        content: str,
        request_id: str,
        is_function_request: bool = False,
    ) -> ChatCompletionResponse:
        """Handle non-streaming chat completion."""

        prompt_tokens = len(esurge.tokenizer(content)["input_ids"])

        sampling_params = self._create_sampling_params(request)
        outputs = esurge.generate(content, sampling_params)

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

        if is_function_request:
            pass

        choices = []
        for idx, completion in enumerate(output.outputs):
            response_text = output.accumulated_text

            if is_function_request:
                format_type = getattr(request, "function_call_format", self.default_function_format)
                parser = FunctionCallParser(format=format_type, strict=False)
                function_calls = parser.parse(response_text)

                if function_calls:
                    message = ChatMessageWithTools.from_function_calls(function_calls, content=None)
                    finish_reason = "tool_calls"
                else:
                    message = ChatMessage(role="assistant", content=response_text)
                    finish_reason = completion.finish_reason or "stop"
            else:
                message = ChatMessage(role="assistant", content=response_text)
                finish_reason = completion.finish_reason or "stop"

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
        is_function_request: bool = False,
    ) -> StreamingResponse:
        """Handle streaming chat completion."""

        async def generate_stream():
            start_time = time.time()
            prompt_tokens = len(esurge.tokenizer(content)["input_ids"])
            sampling_params = self._create_sampling_params(request)

            try:
                total_generated = 0
                first_token_time = None

                initial_chunk = ChatCompletionStreamResponse(
                    model=request.model,
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant"),
                            finish_reason=None,
                        )
                    ],
                    usage=UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        total_tokens=prompt_tokens,
                        tokens_per_second=0.0,
                        processing_time=0.0,
                    ),
                )
                yield f"data: {initial_chunk.model_dump_json(exclude_unset=True, exclude_none=True)}\n\n"
                for output in esurge.stream(content, sampling_params):
                    if first_token_time is None and output.outputs[0].text:
                        first_token_time = time.time() - start_time

                    new_text = output.outputs[0].text
                    if new_text:
                        current_completion_tokens = output.num_generated_tokens
                        current_tps = output.tokens_per_second
                        elapsed_time = output.processing_time
                        if first_token_time is None and output.first_token_time is not None:
                            first_token_time = output.first_token_time

                        chunk = ChatCompletionStreamResponse(
                            model=request.model,
                            choices=[
                                ChatCompletionStreamResponseChoice(
                                    index=0,
                                    delta=DeltaMessage(content=new_text),
                                    finish_reason=None,
                                )
                            ],
                            usage=UsageInfo(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=current_completion_tokens,
                                total_tokens=prompt_tokens + current_completion_tokens,
                                tokens_per_second=current_tps,
                                processing_time=elapsed_time,
                            ),
                        )
                        yield f"data: {chunk.model_dump_json(exclude_unset=True, exclude_none=True)}\n\n"

                        total_generated = output.num_generated_tokens

                    if output.finished:
                        break

                final_output = output
                generation_time = final_output.processing_time
                tokens_per_second = final_output.tokens_per_second
                total_generated = final_output.num_generated_tokens

                usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=total_generated,
                    total_tokens=prompt_tokens + total_generated,
                    tokens_per_second=tokens_per_second,
                    processing_time=generation_time,
                    first_token_time=final_output.first_token_time,
                )

                final_chunk = ChatCompletionStreamResponse(
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
        """Handle completion requests."""
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
        """Handle non-streaming completion."""
        prompt_tokens = len(esurge.tokenizer(prompt)["input_ids"])
        sampling_params = self._create_sampling_params(request)
        outputs = esurge.generate(prompt, sampling_params)

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
        """Handle streaming completion."""

        async def generate_stream():
            sampling_params = self._create_sampling_params(request)
            prompt_tokens = len(esurge.tokenizer(prompt)["input_ids"])

            try:
                total_generated = 0
                first_token_time = None

                for output in esurge.stream(prompt, sampling_params):
                    new_text = output.outputs[0].text
                    if new_text:
                        current_completion_tokens = output.num_generated_tokens
                        current_tps = output.tokens_per_second
                        elapsed_time = output.processing_time
                        if first_token_time is None and output.first_token_time is not None:
                            first_token_time = output.first_token_time

                        chunk = CompletionStreamResponse(
                            model=request.model,
                            choices=[
                                CompletionStreamResponseChoice(
                                    index=0,
                                    text=new_text,
                                    finish_reason=None,
                                )
                            ],
                            usage=UsageInfo(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=current_completion_tokens,
                                total_tokens=prompt_tokens + current_completion_tokens,
                                tokens_per_second=current_tps,
                                processing_time=elapsed_time,
                            ),
                        )
                        yield f"data: {chunk.model_dump_json(exclude_unset=True, exclude_none=True)}\n\n"

                        total_generated = output.num_generated_tokens

                    if output.finished:
                        break

                final_output = output
                generation_time = final_output.processing_time
                tokens_per_second = final_output.tokens_per_second
                total_generated = final_output.num_generated_tokens

                usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=total_generated,
                    total_tokens=prompt_tokens + total_generated,
                    tokens_per_second=tokens_per_second,
                    processing_time=generation_time,
                    first_token_time=final_output.first_token_time,
                )

                final_chunk = CompletionStreamResponse(
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
        """Health check endpoint."""
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
        """Get server performance metrics."""
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

    async def list_tools(self) -> JSONResponse:
        """List available tools/functions for each model."""
        tools_by_model = {}

        for model_name, _ in self.adapters.items():
            model_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "example_function",
                        "description": "An example function for demonstration",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "param1": {
                                    "type": "string",
                                    "description": "First parameter",
                                },
                                "param2": {
                                    "type": "number",
                                    "description": "Second parameter",
                                },
                            },
                            "required": ["param1"],
                        },
                    },
                }
            ]

            tools_by_model[model_name] = {
                "tools": model_tools,
                "formats_supported": ["openai", "hermes", "gorilla", "json_schema"],
                "parallel_calls": True,
            }

        return JSONResponse({"models": tools_by_model, "default_format": "openai"})

    async def _generate_function_followup(
        self,
        request: ChatCompletionRequestWithTools,
        esurge: eSurge,
        executed_calls: list,
        prompt_tokens: int,
        start_time: float,
    ) -> ChatCompletionResponse:
        """Generate a follow-up response with function results."""
        messages = [msg.model_dump() for msg in request.messages]

        function_results = "Function results:\n"
        for call in executed_calls:
            if "result" in call:
                function_results += f"- {call['function']['name']}: {call['result']}\n"
            else:
                function_results += f"- {call['function']['name']}: Error - {call.get('error', 'Unknown error')}\n"

        messages.append(
            {
                "role": "user",
                "content": f"{function_results}\nPlease provide a natural response based on these function results.",
            }
        )

        processor = esurge.tokenizer
        if hasattr(processor, "apply_chat_template"):
            followup_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            followup_prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\nassistant:"

        sampling_params = self._create_sampling_params(request)
        final_outputs = esurge.generate(followup_prompt, sampling_params)

        if not final_outputs:
            raise RuntimeError("Follow-up generation failed")

        final_completion = final_outputs[0].outputs[0]
        final_text = final_completion.text

        completion_tokens = len(final_completion.token_ids)
        generation_time = time.time() - start_time
        tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0

        from ...openai_api_modules import ToolCall

        tool_calls = [ToolCall(id=call["id"], type="function", function=call["function"]) for call in executed_calls]

        message = ChatMessageWithTools(role="assistant", content=final_text, tool_calls=tool_calls)

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
            choices=[
                ChatCompletionResponseChoice(index=0, message=message, finish_reason=completion.finish_reason or "stop")
            ],
            usage=usage,
        )

    async def execute_tool(self, request: tp.Any) -> JSONResponse:
        """Execute a tool/function call.

        This is a placeholder implementation that can be extended
        to integrate with actual tool execution systems.
        """
        return create_error_response(
            HTTPStatus.NOT_IMPLEMENTED,
            "Tool execution endpoint is a placeholder. Implement based on your needs.",
        )
