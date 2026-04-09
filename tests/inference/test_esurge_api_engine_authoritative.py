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

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput, eSurge
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.inference_engine_interface import BaseInferenceApiServer
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    DeltaMessage,
    FunctionCall,
    ResponsesRequest,
    ToolCall,
    UsageInfo,
)
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.stream_protocol import StreamEventFrame
from easydel.inference.typed_models import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseMessageItem,
    ResponseOutputTextPart,
    ResponsesResponse,
    ResponsesUsage,
)


def _make_server() -> eSurgeApiServer:
    server = object.__new__(eSurgeApiServer)
    server.metrics = SimpleNamespace(total_tokens_generated=0, average_tokens_per_second=0.0)
    server._prompt_token_count_from_output = lambda _output: 3
    server._record_api_key_usage = lambda *_args, **_kwargs: None
    return server


def _make_streaming_server() -> eSurgeApiServer:
    server = object.__new__(eSurgeApiServer)
    server.thread_pool = ThreadPoolExecutor(max_workers=1)
    server.metrics = SimpleNamespace(total_tokens_generated=0, average_tokens_per_second=0.0)
    server._active_requests = set()
    server._generation_slots = asyncio.Queue(maxsize=1)
    server._generation_slots.put_nowait(0)
    server._max_generation_slots = 1
    server._overload_message = "busy"
    server._record_api_key_usage = lambda *_args, **_kwargs: None
    server._default_store_responses = False
    server._enable_response_store = False
    return server


async def _collect_streaming_body(response) -> str:
    parts: list[str] = []
    async for chunk in response.body_iterator:
        parts.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(parts)


def _make_request(with_tools: bool = False) -> ChatCompletionRequest:
    payload = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    if with_tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                    },
                },
            }
        ]
    return ChatCompletionRequest.model_validate(payload)


def test_build_chat_completion_response_keeps_raw_text_when_engine_has_no_tool_calls():
    server = _make_server()
    server.extract_tool_calls_batch = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("API server fallback parser should not run")
    )
    request = _make_request(with_tools=True)
    raw_text = "<tool_call><function=lookup><parameter=query>AI</parameter></function></tool_call>"
    output = RequestOutput(
        request_id="req_1",
        prompt="hi",
        prompt_token_ids=[1, 2, 3],
        outputs=[
            CompletionOutput(
                index=0,
                text=raw_text,
                token_ids=[10, 11],
                finish_reason="stop",
            )
        ],
        accumulated_text=raw_text,
        num_generated_tokens=5,
        tokens_per_second=1.0,
        processing_time=2.0,
        first_token_time=0.1,
    )

    response = server._build_chat_completion_response(request, SimpleNamespace(), output, None)

    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == raw_text
    assert response.choices[0].message.tool_calls is None
    assert response.usage.prompt_tokens == 3


def test_build_chat_completion_response_uses_engine_tool_calls_without_reparsing():
    server = _make_server()
    server.extract_tool_calls_batch = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("API server fallback parser should not run")
    )
    request = _make_request(with_tools=True)
    output = RequestOutput(
        request_id="req_2",
        prompt="hi",
        prompt_token_ids=[1, 2, 3],
        outputs=[
            CompletionOutput(
                index=0,
                text="ignored raw text",
                token_ids=[10, 11],
                finish_reason="stop",
                tool_calls=[
                    ToolCall(
                        type="function",
                        function=FunctionCall(name="lookup", arguments='{"query":"AI"}'),
                    )
                ],
            )
        ],
        accumulated_text="ignored raw text",
        num_generated_tokens=5,
        tokens_per_second=1.0,
        processing_time=2.0,
        first_token_time=0.1,
    )

    response = server._build_chat_completion_response(request, SimpleNamespace(), output, None)

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.content is None
    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].message.tool_calls[0].function.name == "lookup"


def test_list_tools_reflects_engine_tool_parser_configuration():
    server = object.__new__(eSurgeApiServer)
    server._authorize_request = lambda _raw_request: None
    server.enable_function_calling = True
    server.adapters = {
        "dummy-model": SimpleNamespace(esurge=SimpleNamespace(tool_parser="qwen3_xml")),
    }

    response = asyncio.run(server.list_tools(SimpleNamespace()))
    payload = json.loads(response.body)

    assert payload["models"]["dummy-model"]["tool_parser"] == "qwen3_xml"
    assert payload["models"]["dummy-model"]["formats_supported"] == ["qwen3_xml"]


def test_esurge_auto_detect_tool_parser_uses_tokenizer_hints_without_model_type():
    class DummyTokenizer:
        chat_template = '<assistant><function name="lookup">'

        @staticmethod
        def get_vocab():
            return {}

    detected = eSurge._auto_detect_tool_parser(
        tokenizer=DummyTokenizer(),
        model_type=None,
    )

    assert detected == "qwen3_xml"


def test_handle_chat_streaming_forwards_engine_chunks_without_api_delta_logic():
    server = _make_streaming_server()
    server._prepare_sampling_params = lambda request, esurge: SamplingParams(max_tokens=8)
    server._compute_delta_text = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("API delta recomputation should not run")
    )
    server._coerce_stream_delta_message = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("API delta coercion should not run")
    )

    async def _is_disconnected():
        return False

    class FakeSurge:
        def abort_request(self, request_id):
            raise AssertionError(f"abort_request should not be called: {request_id}")

        def iter_chat_completion_stream(self, **kwargs):
            assert kwargs["model"] == "dummy-model"
            assert kwargs["request_id"] == "req-stream"
            return iter(
                [
                    ChatCompletionStreamResponse(
                        model="dummy-model",
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(role="assistant", content="Hello!"),
                                finish_reason=None,
                            )
                        ],
                        usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
                    ),
                    ChatCompletionStreamResponse(
                        model="dummy-model",
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(role="assistant", content=""),
                                finish_reason="stop",
                            )
                        ],
                        usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
                    ),
                ]
            )

    request = _make_request()
    raw_request = SimpleNamespace(is_disconnected=_is_disconnected)

    try:
        response = asyncio.run(
            server._handle_chat_streaming(
                request,
                FakeSurge(),
                messages=[{"role": "user", "content": "hi"}],
                request_id="req-stream",
                raw_request=raw_request,
            )
        )
        body = asyncio.run(_collect_streaming_body(response))
    finally:
        server.thread_pool.shutdown(wait=True)

    expected_chunks = [
        ChatCompletionStreamResponse(
            model="dummy-model",
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content="Hello!"),
                    finish_reason=None,
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        ),
        ChatCompletionStreamResponse(
            model="dummy-model",
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        ),
    ]
    assert body == "".join(
        [
            f"data: {expected_chunks[0].model_dump_json(exclude_unset=True, exclude_none=True)}\n\n",
            f"data: {expected_chunks[1].model_dump_json(exclude_unset=True, exclude_none=True)}\n\n",
            "data: [DONE]\n\n",
        ]
    )


def test_responses_streaming_forwards_engine_frames_without_api_parser_state():
    server = _make_streaming_server()
    server._extract_payload_api_keys = lambda request: []
    server._authorize_request = lambda *args, **kwargs: None
    server._parse_responses_max_tokens = lambda payload, esurge: (8, 8)
    server._responses_payload_to_messages = lambda payload, include_instructions=False: [
        {"role": "user", "content": "ping"}
    ]
    server._extract_responses_tools = lambda payload: (None, None)
    server._create_sampling_params_from_responses = lambda payload, max_tokens: SamplingParams(max_tokens=max_tokens)
    server._apply_extra_stops_to_sampling_params = lambda sampling_params: sampling_params
    server._responses_reasoning_summary_requested = lambda payload: False
    server._compute_delta_text = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("API delta recomputation should not run")
    )
    server._jsonify_tool_calls = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("API tool-call shaping should not run during streaming")
    )

    async def _is_disconnected():
        return False

    final_obj = ResponsesResponse(
        id="resp_test",
        object="response",
        created_at=123,
        model="demo-model",
        status="completed",
        output=[
            ResponseMessageItem(
                id="msg_1",
                role="assistant",
                content=[ResponseOutputTextPart(text="pong")],
                status="completed",
            )
        ],
        usage=ResponsesUsage(input_tokens=2, output_tokens=1, total_tokens=3),
        store=False,
    )

    class FakeSurge:
        def abort_request(self, request_id):
            raise AssertionError(f"abort_request should not be called: {request_id}")

        def iter_responses_stream(self, **kwargs):
            assert kwargs["response_id"].startswith("resp_")
            return iter(
                [
                    StreamEventFrame(
                        event="response.created",
                        payload=ResponseCreatedEvent(
                            response=ResponsesResponse(
                                id=kwargs["response_id"],
                                object="response",
                                created_at=123,
                                model=kwargs["model"],
                                status="in_progress",
                                output=[],
                            )
                        ),
                    ),
                    StreamEventFrame(
                        event="response.completed",
                        payload=ResponseCompletedEvent(response=final_obj),
                    ),
                ]
            )

    server._get_adapter = lambda model: SimpleNamespace(esurge=FakeSurge())
    request = ResponsesRequest.model_validate({"model": "demo-model", "input": "ping", "stream": True, "store": False})
    raw_request = SimpleNamespace(is_disconnected=_is_disconnected)

    try:
        response = asyncio.run(server.responses(request, raw_request))
        body = asyncio.run(_collect_streaming_body(response))
    finally:
        server.thread_pool.shutdown(wait=True)

    response_id = next(
        line.split('"id":"', 1)[1].split('"', 1)[0] for line in body.splitlines() if '"type":"response.created"' in line
    )
    expected_created = BaseInferenceApiServer._sse_event(
        "response.created",
        ResponseCreatedEvent(
            response=ResponsesResponse(
                id=response_id,
                object="response",
                created_at=123,
                model="demo-model",
                status="in_progress",
                output=[],
            )
        ),
    )
    expected_completed = BaseInferenceApiServer._sse_event(
        "response.completed",
        ResponseCompletedEvent(response=final_obj),
    )

    assert body == expected_created + expected_completed
