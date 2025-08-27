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

"""Enhanced FastAPI server for serving vInference models with OpenAI API compatibility."""

from __future__ import annotations

import asyncio
import time
import traceback
import typing as tp
from http import HTTPStatus

from eformer.loggings import get_logger
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import ProcessorMixin

from ..inference_engine_interface import (
    BaseInferenceApiServer,
    InferenceEngineAdapter,
    ServerStatus,
    create_error_response,
)
from ..openai_api_modules import (
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
    ToolCall,
    UsageInfo,
)
from ..sampling_params import SamplingParams

if tp.TYPE_CHECKING:
    from ..vinference import vInference

logger = get_logger("vInferenceApiServer")


class vInferenceAdapter(InferenceEngineAdapter):
    """Adapter for vInference engine."""

    def __init__(self, vinference_instance: vInference):
        """Initialize vInference adapter."""
        from ..vinference import vInference

        if not isinstance(vinference_instance, vInference):
            raise TypeError(f"Expected vInference instance, got {type(vinference_instance).__name__}")

        self.vinference = vinference_instance

    async def generate(
        self,
        prompts: str | list[str] | dict,
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> tp.Any | tp.AsyncGenerator[tp.Any, None]:
        """Generate using vInference."""
        ids = prompts  # Already tokenized

        if stream:
            return self.vinference.generate(**ids, sampling_params=sampling_params)
        else:
            response_state = None
            for response_state in self.vinference.generate(**ids, sampling_params=sampling_params):  # noqa
                pass
            return response_state

    def count_tokens(self, content: str) -> int:
        """Count tokens using vInference tokenizer."""
        tokens = self.vinference.tokenizer(content, return_tensors="np")
        return tokens["input_ids"].shape[-1]

    def get_model_info(self) -> dict[str, tp.Any]:
        """Get vInference model information."""
        return {
            "name": self.vinference.inference_name,
            "type": "vinference",
            "architecture": type(self.vinference).__name__,
            "supports_streaming": hasattr(self.vinference.generation_config, "streaming_chunks"),
            "max_tokens": getattr(self.vinference.generation_config, "max_new_tokens", None),
        }

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.vinference.inference_name

    @property
    def processor(self) -> tp.Any:
        """Get processor/tokenizer."""
        return self.vinference.processor_class

    @property
    def tokenizer(self) -> tp.Any:
        """Get tokenizer."""
        return self.vinference.tokenizer


class vInferenceApiServer(BaseInferenceApiServer):
    """
    Enhanced FastAPI server for serving vInference instances.

    Inherits from BaseInferenceApiServer for consistent API structure.
    """

    def __init__(
        self,
        inference_map: dict[str, vInference] | vInference | None = None,
        inference_init_call: tp.Callable[[], dict[str, vInference] | vInference] | None = None,
        allow_parallel_workload: bool = False,
        oai_like_processor: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the vInference API Server.

        Args:
            inference_map: Dictionary of model names to vInference instances or single instance
            inference_init_call: Callable for lazy initialization of models
            allow_parallel_workload: Allow parallel request processing
            oai_like_processor: Use OpenAI-like conversation format
            **kwargs: Additional arguments for base server
        """
        from ..vinference import vInference

        # Initialize models
        if inference_init_call is not None:
            inference_map = inference_init_call()

        if inference_map is None:
            raise ValueError("Either `inference_map` or `inference_init_call` must be provided.")

        if isinstance(inference_map, vInference):
            inference_map = {inference_map.inference_name: inference_map}

        # Create adapters for each vInference instance
        self.adapters: dict[str, vInferenceAdapter] = {}
        for name, inference in inference_map.items():
            if not isinstance(inference, vInference):
                raise TypeError(f"Value for key '{name}' must be an instance of vInference")
            self.adapters[name] = vInferenceAdapter(inference)

        self.allow_parallel_workload = allow_parallel_workload
        self.oai_like_processor = oai_like_processor

        # Initialize base server
        super().__init__(
            server_name="EasyDeL vInference API Server",
            server_description="High-performance vInference server with OpenAI API compatibility",
            **kwargs,
        )

    async def on_startup(self) -> None:
        """Custom startup logic for vInference."""
        logger.info(f"Loaded {len(self.adapters)} vInference models")
        for name in self.adapters:
            logger.info(f"  - {name}")

    def _get_adapter(self, model_name: str) -> vInferenceAdapter:
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
        # Use first available model if no specific model requested
        adapter = next(iter(self.adapters.values()))
        return adapter.count_tokens(content)

    def _create_sampling_params(self, request: ChatCompletionRequest | CompletionRequest) -> SamplingParams:
        """Create sampling parameters from request."""
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
            n=int(request.n or 1),
            stop=request.stop,
        )

    def _prepare_tokenized_input(
        self,
        request: ChatCompletionRequest,
        adapter: vInferenceAdapter,
    ) -> dict:
        """Prepare and tokenize chat input."""
        messages = [msg.model_dump() for msg in request.messages]
        processor = adapter.processor

        if isinstance(processor, ProcessorMixin) and self.oai_like_processor:
            from easydel.trainers.prompt_utils import convert_to_openai_format

            messages = convert_to_openai_format(messages)

        try:
            if request.chat_template_kwargs is None:
                request.chat_template_kwargs = {}
            add_generation_prompt = request.chat_template_kwargs.pop("add_generation_prompt", True)

            # First, apply chat template to get text
            text = processor.apply_chat_template(
                conversation=messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,  # Don't tokenize yet
                **request.chat_template_kwargs,
            )

            # Then tokenize the text
            tokenized = adapter.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                return_attention_mask=True,
            )

            return tokenized

        except Exception as e:
            logger.exception(f"Error applying chat template: {e}")
            raise RuntimeError(f"Error tokenizing input: {e}") from e

    async def _prepare_tokenized_input_async(self, request: ChatCompletionRequest, adapter: vInferenceAdapter) -> dict:
        """Prepare tokenized input asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._prepare_tokenized_input,
            request,
            adapter,
        )

    async def chat_completions(self, request: ChatCompletionRequest) -> tp.Any:
        """Handle chat completion requests with function calling support."""
        request_id = getattr(request, "request_id", None)

        try:
            if not request.messages:
                raise HTTPException(400, "Messages cannot be empty")

            adapter = self._get_adapter(request.model)

            # Check if this is a function calling request
            is_function_request = self.enable_function_calling and request.get_tools()

            ids = await self._prepare_tokenized_input_async(request, adapter)

            # Generate response
            if request.stream:
                return await self._handle_streaming_response(request, adapter, ids, request_id, is_function_request)
            else:
                return await self._handle_non_streaming_response(request, adapter, ids, request_id, is_function_request)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            logger.exception(f"Error in chat completion: {e}")
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def completions(self, request: CompletionRequest) -> tp.Any:
        """Handle completion requests."""
        request_id = getattr(request, "request_id", None)

        try:
            adapter = self._get_adapter(request.model)

            prompt = request.prompt
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""

            if not prompt:
                raise HTTPException(400, "Prompt cannot be empty")

            # Tokenize prompt
            inputs = await self._tokenize_prompt_async(adapter, prompt)

            # Generate response
            if request.stream:
                return await self._handle_completion_streaming(request, adapter, inputs, request_id)
            else:
                return await self._handle_completion_response(request, adapter, inputs, request_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in completion: {e}")

            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), request_id)

    async def _tokenize_prompt_async(self, adapter: vInferenceAdapter, prompt: str) -> dict:
        """Tokenize a prompt asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: adapter.tokenizer(prompt, return_tensors="np", padding=True),
        )

    async def _handle_non_streaming_response(
        self,
        request: ChatCompletionRequest,
        adapter: vInferenceAdapter,
        ids: dict,
        request_id: str | None = None,
        is_function_request: bool = False,
    ) -> ChatCompletionResponse:
        """Generate non-streaming response with function calling support."""
        start_time = time.time()

        try:
            prompt_tokens = ids["input_ids"].shape[-1]
            sampling_params = self._create_sampling_params(request)

            # Generate response
            response_state = await adapter.generate(ids, sampling_params, stream=False)

            if response_state is None:
                raise RuntimeError("Generation failed to produce output")

            # Decode generated text
            final_sequences = response_state.sequences
            generated_tokens = response_state.generated_tokens
            padded_length = response_state.padded_length

            final_responses = adapter.tokenizer.batch_decode(
                final_sequences[..., padded_length:],
                skip_special_tokens=True,
            )

            # Update metrics
            self.metrics.total_tokens_generated += generated_tokens
            generation_time = time.time() - start_time
            tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0

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
                    message = ChatMessage(role="assistant", content=response_text)
                    finish_reason = self._determine_finish_reason(
                        generated_tokens, sampling_params.max_tokens, response_text
                    )
                else:
                    message = ChatMessage(role="assistant", content=response_text)
                    finish_reason = self._determine_finish_reason(
                        generated_tokens, sampling_params.max_tokens, response_text
                    )

                choices.append(
                    ChatCompletionResponseChoice(
                        index=i,
                        message=message,
                        finish_reason=finish_reason,
                    )
                )

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

    async def _handle_streaming_response(
        self,
        request: ChatCompletionRequest,
        adapter: vInferenceAdapter,
        ids: dict,
        request_id: str | None = None,
        is_function_request: bool = False,
    ) -> StreamingResponse:
        """Generate streaming response with function calling support."""

        async def generate_stream():
            start_time = time.time()
            prompt_tokens = ids["input_ids"].shape[-1]
            sampling_params = self._create_sampling_params(request)

            parser = None
            try:
                padded_length = None
                current_position = 0
                total_generated = 0
                first_token_time = None
                accumulated_text = [""] * ids["input_ids"].shape[0]
                partial_function_calls = [[] for _ in range(ids["input_ids"].shape[0])]

                # Get the generator from adapter
                generator = await adapter.generate(ids, sampling_params, stream=True)

                # If it's a sync generator, we need to wrap it for async iteration
                if not hasattr(generator, "__aiter__"):

                    async def async_wrapper():
                        for item in generator:
                            yield item
                            # Allow other coroutines to run
                            await asyncio.sleep(0)

                    async for response_state in async_wrapper():
                        padded_length = response_state.padded_length

                        if current_position == 0:
                            current_position = padded_length

                        new_position = padded_length + response_state.generated_tokens
                        if new_position > current_position:
                            new_tokens = response_state.sequences[..., current_position:new_position]
                            decoded_chunks = adapter.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                            if first_token_time is None and response_state.generated_tokens > 0:
                                first_token_time = time.time() - start_time

                            tokens_generated = new_position - current_position
                            total_generated += tokens_generated
                            self.metrics.total_tokens_generated += tokens_generated

                            choices = []
                            for i, chunk_text in enumerate(decoded_chunks):
                                accumulated_text[i] += chunk_text

                                if is_function_request and parser:
                                    # Try to parse function calls
                                    function_calls = parser.parse(accumulated_text[i])
                                    if function_calls and len(function_calls) > len(partial_function_calls[i]):
                                        new_call = function_calls[-1]
                                        partial_function_calls[i].append(new_call)

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

                            if choices:
                                chunk = ChatCompletionStreamResponse(
                                    model=request.model,
                                    choices=choices,
                                    usage=UsageInfo(),
                                )
                                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                            current_position = new_position
                else:
                    # It's already an async generator
                    async for response_state in generator:
                        # Same processing logic as above
                        padded_length = response_state.padded_length

                        if current_position == 0:
                            current_position = padded_length

                        new_position = padded_length + response_state.generated_tokens
                        if new_position > current_position:
                            new_tokens = response_state.sequences[..., current_position:new_position]
                            decoded_chunks = adapter.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                            if first_token_time is None and response_state.generated_tokens > 0:
                                first_token_time = time.time() - start_time

                            tokens_generated = new_position - current_position
                            total_generated += tokens_generated
                            self.metrics.total_tokens_generated += tokens_generated

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

                            if choices:
                                chunk = ChatCompletionStreamResponse(
                                    model=request.model,
                                    choices=choices,
                                    usage=UsageInfo(),
                                )
                                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                            current_position = new_position

                # Send final chunk
                if total_generated > 0:
                    generation_time = time.time() - start_time
                    tokens_per_second = total_generated / generation_time if generation_time > 0 else 0

                    final_choices = []
                    for i in range(len(accumulated_text)):
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

    async def _handle_completion_response(
        self,
        request: CompletionRequest,
        adapter: vInferenceAdapter,
        inputs: dict,
        request_id: str | None = None,
    ) -> CompletionResponse:
        """Generate non-streaming completion response."""
        start_time = time.time()

        try:
            prompt_tokens = inputs["input_ids"].shape[-1]
            sampling_params = self._create_sampling_params(request)

            response_state = await adapter.generate(inputs, sampling_params, stream=False)

            if response_state is None:
                raise RuntimeError("Generation failed to produce output")

            completions = adapter.tokenizer.batch_decode(
                response_state.sequences[..., response_state.padded_length :],
                skip_special_tokens=True,
            )

            generated_tokens = response_state.generated_tokens
            self.metrics.total_tokens_generated += generated_tokens
            generation_time = time.time() - start_time

            choices = [
                CompletionResponseChoice(
                    text=completion,
                    index=i,
                    finish_reason=self._determine_finish_reason(
                        generated_tokens, sampling_params.max_tokens, completion
                    ),
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

    async def _handle_completion_streaming(
        self,
        request: CompletionRequest,
        adapter: vInferenceAdapter,
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

                # Get the generator from adapter
                generator = await adapter.generate(inputs, sampling_params, stream=True)

                # Wrap sync generator for async iteration
                async def async_wrapper():
                    for item in generator:
                        yield item
                        await asyncio.sleep(0)

                async for response_state in async_wrapper():
                    padded_length = response_state.padded_length

                    if current_position == 0:
                        current_position = padded_length

                    new_position = padded_length + response_state.generated_tokens
                    if new_position > current_position:
                        new_tokens = response_state.sequences[..., current_position:new_position]
                        decoded_chunks = adapter.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

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

                        chunk = CompletionStreamResponse(
                            model=request.model,
                            choices=choices,
                        )

                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                        current_position = new_position

                # Send final chunk
                if total_generated > 0:
                    generation_time = time.time() - start_time
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
                        tokens_per_second=total_generated / generation_time if generation_time > 0 else 0,
                        processing_time=generation_time,
                        first_token_time=first_token_time,
                    )

                    final_chunk = CompletionStreamResponse(
                        model=request.model,
                        choices=final_choices,
                        usage=usage,
                    )

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
        for name, adapter in self.adapters.items():
            try:
                model_info = adapter.get_model_info()
                health_status["models"][name] = {
                    "loaded": True,
                    **model_info,
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

    async def get_metrics(self) -> JSONResponse:
        """Get detailed server performance metrics."""
        return JSONResponse(
            {
                "uptime_seconds": round(self.metrics.uptime_seconds, 2),
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": round(self.metrics.successful_requests / max(self.metrics.total_requests, 1) * 100, 2),
                "total_tokens_generated": self.metrics.total_tokens_generated,
                "average_tokens_per_second": round(self.metrics.average_tokens_per_second, 2),
                "active_requests": len(self._active_requests),
                "models": {
                    "loaded": len(self.adapters),
                    "names": list(self.adapters.keys()),
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

    async def list_models(self) -> JSONResponse:
        """List available models with detailed information."""
        models_data = []

        for model_id, adapter in self.adapters.items():
            try:
                model_info = adapter.get_model_info()
                models_data.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": int(self.metrics.start_time),
                        "owned_by": "easydel",
                        "permission": [],
                        "root": model_id,
                        "parent": None,
                        "metadata": {
                            **model_info,
                            "supports_chat": hasattr(adapter.processor, "apply_chat_template"),
                            "supports_function_calling": self.enable_function_calling,
                        },
                    }
                )
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
            adapter = self._get_adapter(model_id)
            model_info = adapter.get_model_info()

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
                        **model_info,
                        "supports_chat": hasattr(adapter.processor, "apply_chat_template"),
                        "supports_function_calling": self.enable_function_calling,
                    },
                }
            )
        except HTTPException:
            raise
        except Exception as e:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error retrieving model info: {e!s}")

    async def list_tools(self) -> JSONResponse:
        """List available tools/functions for each model."""
        tools_by_model = {}

        return JSONResponse({"models": tools_by_model, "default_format": self.default_function_format.value})

    async def execute_tool(self, request: Request) -> JSONResponse:
        """Execute a tool/function call."""
        try:
            body = await request.json()
            tool_name = body.get("name")
            tool_args = body.get("arguments", {})

            # This is a placeholder implementation
            # In a real scenario, you would:
            # 1. Validate the tool exists
            # 2. Execute the actual function
            # 3. Return the results

            # Example mock responses
            if tool_name == "get_weather":
                location = tool_args.get("location", "Unknown")
                unit = tool_args.get("unit", "celsius")
                return JSONResponse(
                    {
                        "result": {
                            "location": location,
                            "temperature": 22 if unit == "celsius" else 72,
                            "unit": unit,
                            "condition": "Partly cloudy",
                        },
                        "success": True,
                    }
                )
            elif tool_name == "search_web":
                query = tool_args.get("query", "")
                num_results = tool_args.get("num_results", 5)
                return JSONResponse(
                    {
                        "result": {
                            "query": query,
                            "results": [
                                {
                                    "title": f"Result {i + 1} for '{query}'",
                                    "url": f"https://example.com/result{i + 1}",
                                    "snippet": f"This is a snippet for result {i + 1}",
                                }
                                for i in range(min(num_results, 5))
                            ],
                        },
                        "success": True,
                    }
                )
            else:
                return JSONResponse(
                    {
                        "error": f"Unknown tool: {tool_name}",
                        "success": False,
                    },
                    status_code=404,
                )

        except Exception as e:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f"Tool execution failed: {e!s}")

    # Remove the run() method - use parent's implementation
