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

from types import SimpleNamespace

from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.openai_api_modules import ChatCompletionRequest, FunctionCall, ToolCall


def _make_server() -> eSurgeApiServer:
    server = object.__new__(eSurgeApiServer)
    server.metrics = SimpleNamespace(total_tokens_generated=0, average_tokens_per_second=0.0)
    server._prompt_token_count_from_output = lambda _output: 3
    server._record_api_key_usage = lambda *_args, **_kwargs: None
    return server


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
    assert response.choices[0].message.content == ""
    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].message.tool_calls[0].function.name == "lookup"
