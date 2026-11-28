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

"""Pure-Python FastAPI contract tests for Anthropic Messages endpoints on eSurgeApiServer."""

from __future__ import annotations

import json
import typing as tp

import pytest
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.openai_api_modules import (
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from fastapi.testclient import TestClient

pytestmark = pytest.mark.cpu_ok


class _FakeTokenizer:
    chat_template = "{% if enable_thinking %}{% if reasoning_effort == 'xhigh' %}{% endif %}{% endif %}{% if preserve_thinking %}{% endif %}"

    def __call__(self, text: str) -> dict[str, list[int]]:
        toks = text.split()
        return {"input_ids": list(range(max(1, len(toks))))}

    def apply_chat_template(
        self,
        messages: list[dict[str, tp.Any]],
        *,
        tools: list[dict[str, tp.Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: tp.Any,
    ) -> str:
        del tokenize, add_generation_prompt, kwargs
        parts = [f"{m.get('role')}:{m.get('content', '')}" for m in messages]
        if tools:
            parts.append(f"tools:{len(tools)}")
        return " ".join(parts)


class _FakeSurge:
    def __init__(self, *, mode: str = "tool_and_thinking"):
        self.esurge_name = "test-model"
        self.tokenizer = _FakeTokenizer()
        self.max_model_len = 32768
        self.max_num_seqs = 16
        self.mode = mode
        self.last_chat_kwargs: dict[str, tp.Any] | None = None
        self.aborted: list[str] = []

    def generate(self, *args, **kwargs):
        del args, kwargs
        return []

    def stream(self, *args, **kwargs):
        del args, kwargs
        return iter(())

    def abort_request(self, request_id: str) -> None:
        self.aborted.append(request_id)

    def chat(
        self,
        *,
        messages: list[dict[str, tp.Any]],
        tools: list[dict[str, tp.Any]] | None = None,
        tool_choice: tp.Any = None,
        sampling_params: tp.Any = None,
        request_id: str | None = None,
        stream: bool = False,
        chat_template_kwargs: dict[str, tp.Any] | None = None,
    ) -> RequestOutput:
        self.last_chat_kwargs = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "sampling_params": sampling_params,
            "stream": stream,
            "chat_template_kwargs": chat_template_kwargs,
        }
        if self.mode == "tool_and_thinking":
            comp = CompletionOutput(
                index=0,
                text="Running bash now.",
                raw_text="Running bash now.<tool_call><function=Bash></function></tool_call>",
                token_ids=[1, 2, 3, 4],
                finish_reason="tool_calls",
                reasoning_content="Need to inspect files first.",
                tool_calls=[
                    ToolCall(
                        id="toolu_test123",
                        type="function",
                        function=FunctionCall(name="Bash", arguments='{"command": "ls -la"}'),
                    )
                ],
            )
        elif self.mode == "max_tokens":
            comp = CompletionOutput(
                index=0,
                text="Truncated output",
                token_ids=[1, 2],
                finish_reason="length",
            )
        else:
            comp = CompletionOutput(
                index=0,
                text="All done!",
                token_ids=[1, 2, 3],
                finish_reason="stop",
                reasoning_content="Verified result.",
            )
        return RequestOutput(
            request_id=request_id or "req-1",
            prompt="prompt",
            prompt_token_ids=[10, 11, 12, 13, 14],
            outputs=[comp],
            finished=True,
            accumulated_text=comp.text,
            num_generated_tokens=len(comp.token_ids),
            metrics={"cache_read": 2, "cache_written": 3},
            tokens_per_second=42.0,
        )

    def iter_chat_completion_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, tp.Any]],
        tools: list[dict[str, tp.Any]] | None = None,
        tool_choice: tp.Any = None,
        sampling_params: tp.Any = None,
        request_id: str | None = None,
        chat_template_kwargs: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ):
        del kwargs
        self.last_chat_kwargs = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "sampling_params": sampling_params,
            "chat_template_kwargs": chat_template_kwargs,
        }
        yield ChatCompletionStreamResponse(
            id=request_id or "chatcmpl-1",
            model=model,
            choices=[ChatCompletionStreamResponseChoice(index=0, delta=DeltaMessage(reasoning_content="Let me "))],
            usage=UsageInfo(prompt_tokens=25, completion_tokens=1, total_tokens=26),
        )
        yield ChatCompletionStreamResponse(
            id=request_id or "chatcmpl-1",
            model=model,
            choices=[ChatCompletionStreamResponseChoice(index=0, delta=DeltaMessage(reasoning_content="check."))],
            usage=UsageInfo(prompt_tokens=25, completion_tokens=2, total_tokens=27),
        )
        if self.mode == "tool_and_thinking":
            yield ChatCompletionStreamResponse(
                id=request_id or "chatcmpl-1",
                model=model,
                choices=[ChatCompletionStreamResponseChoice(index=0, delta=DeltaMessage(content="Calling tool."))],
                usage=UsageInfo(prompt_tokens=25, completion_tokens=4, total_tokens=29),
            )
            yield ChatCompletionStreamResponse(
                id=request_id or "chatcmpl-1",
                model=model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=0,
                                    id="toolu_stream1",
                                    type="function",
                                    function=DeltaFunctionCall(name="Bash", arguments='{"command":'),
                                )
                            ]
                        ),
                    )
                ],
                usage=UsageInfo(prompt_tokens=25, completion_tokens=6, total_tokens=31),
            )
            yield ChatCompletionStreamResponse(
                id=request_id or "chatcmpl-1",
                model=model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=0,
                                    function=DeltaFunctionCall(arguments=' "echo hi"}'),
                                )
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=UsageInfo(prompt_tokens=25, completion_tokens=8, total_tokens=33),
            )
        else:
            yield ChatCompletionStreamResponse(
                id=request_id or "chatcmpl-1",
                model=model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(content="Hello world!"),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(prompt_tokens=25, completion_tokens=5, total_tokens=30),
            )


def _parse_sse(raw_text: str) -> list[tuple[str, dict[str, tp.Any]]]:
    events: list[tuple[str, dict[str, tp.Any]]] = []
    cur_event = ""
    for line in raw_text.splitlines():
        if line.startswith("event:"):
            cur_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((cur_event, json.loads(line.split(":", 1)[1].strip())))
    return events


def test_anthropic_messages_non_streaming_thinking_tools_and_multi_turn():
    fake = _FakeSurge(mode="tool_and_thinking")
    server = eSurgeApiServer({"test-model": fake}, enable_persistence=False)
    client = TestClient(server.app)

    resp = client.post(
        "/v1/messages?beta=true",
        json={
            "model": "test-model",
            "max_tokens": 2048,
            "system": [{"type": "text", "text": "You are a coding assistant."}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
                }
            ],
            "tool_choice": {"type": "any"},
            "messages": [
                {"role": "user", "content": "Inspect the repo"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Prior thought", "signature": "sig_prev"},
                        {"type": "tool_use", "id": "toolu_0", "name": "Bash", "input": {"command": "pwd"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_0", "content": "/testbed"},
                        {"type": "text", "text": "Now list files."},
                    ],
                },
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "test-model"
    assert body["stop_reason"] == "tool_use"
    assert body["usage"]["input_tokens"] == 5
    assert body["usage"]["output_tokens"] == 4
    assert body["usage"]["cache_read_input_tokens"] == 2

    types = [b["type"] for b in body["content"]]
    assert types == ["thinking", "text", "tool_use"]
    assert body["content"][0]["thinking"] == "Need to inspect files first."
    assert body["content"][0]["signature"]
    assert body["content"][1]["text"] == "Running bash now."
    assert body["content"][2] == {
        "type": "tool_use",
        "id": "toolu_test123",
        "name": "Bash",
        "input": {"command": "ls -la"},
    }

    assert fake.last_chat_kwargs is not None
    msgs = fake.last_chat_kwargs["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]
    assert msgs[2]["reasoning_content"] == "Prior thought"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "Bash"
    assert msgs[3]["tool_call_id"] == "toolu_0"
    assert msgs[3]["content"] == "/testbed"
    assert fake.last_chat_kwargs["tool_choice"] == "required"
    assert fake.last_chat_kwargs["chat_template_kwargs"]["enable_thinking"] is True
    assert fake.last_chat_kwargs["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    assert fake.last_chat_kwargs["chat_template_kwargs"]["preserve_thinking"] is True


def test_anthropic_messages_streaming_sse_contract():
    fake = _FakeSurge(mode="tool_and_thinking")
    server = eSurgeApiServer({"test-model": fake}, enable_persistence=False)
    client = TestClient(server.app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 1024,
            "stream": True,
            "messages": [{"role": "user", "content": "Say hi via Bash"}],
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    names = [ev for ev, _ in events]

    assert names[0] == "message_start"
    assert names[-2] == "message_delta"
    assert names[-1] == "message_stop"

    thinking_deltas = [
        d["delta"]["thinking"]
        for ev, d in events
        if ev == "content_block_delta" and d["delta"]["type"] == "thinking_delta"
    ]
    assert "".join(thinking_deltas) == "Let me check."
    assert any(ev == "content_block_delta" and d["delta"]["type"] == "signature_delta" for ev, d in events)
    text_deltas = [
        d["delta"]["text"] for ev, d in events if ev == "content_block_delta" and d["delta"]["type"] == "text_delta"
    ]
    assert "".join(text_deltas) == "Calling tool."
    tool_starts = [
        d["content_block"]
        for ev, d in events
        if ev == "content_block_start" and d["content_block"]["type"] == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0]["id"] == "toolu_stream1"
    assert tool_starts[0]["name"] == "Bash"
    json_deltas = [
        d["delta"]["partial_json"]
        for ev, d in events
        if ev == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
    ]
    assert json.loads("".join(json_deltas)) == {"command": "echo hi"}
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"
    assert events[-2][1]["usage"]["output_tokens"] == 8


def test_anthropic_count_tokens_and_auth_and_error_envelopes():
    fake = _FakeSurge(mode="max_tokens")
    server = eSurgeApiServer({"test-model": fake}, require_api_key=True, enable_persistence=False)
    server.auth_manager.create_api_key(raw_key="sk-ant-test-1234567890", name="ant", created_by="test")
    client = TestClient(server.app)

    # Missing x-api-key returns official Anthropic authentication_error envelope.
    unauth = client.post(
        "/v1/messages",
        json={"model": "test-model", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert unauth.status_code == 401
    assert unauth.json()["type"] == "error"
    assert unauth.json()["error"]["type"] == "authentication_error"

    # Unknown model returns not_found_error.
    not_found = client.post(
        "/v1/messages",
        headers={"x-api-key": "sk-ant-test-1234567890"},
        json={"model": "missing-model", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert not_found.status_code == 404
    assert not_found.json()["error"]["type"] == "not_found_error"

    # Authorized count_tokens returns exact token count.
    counted = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "sk-ant-test-1234567890"},
        json={
            "model": "test-model",
            "system": "Be concise",
            "messages": [{"role": "user", "content": "one two three four"}],
        },
    )
    assert counted.status_code == 200
    assert counted.json()["input_tokens"] >= 5

    # Authorized max_tokens stop_reason mapping.
    ok = client.post(
        "/v1/messages",
        headers={"x-api-key": "sk-ant-test-1234567890"},
        json={"model": "test-model", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert ok.status_code == 200
    assert ok.json()["stop_reason"] == "max_tokens"
