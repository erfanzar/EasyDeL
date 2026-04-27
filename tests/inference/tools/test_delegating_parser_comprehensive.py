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

"""Comprehensive tests for the DelegatingParser state machine."""

from dataclasses import dataclass

from easydel.inference.openai_api_modules import ChatCompletionRequest, ChatMessage
from easydel.inference.parsing.delegating_parser import (
    DelegatingParser,
    ParsePhase,
    ParseResult,
)


class _FakeReasoningParser:
    """Simulates a reasoning parser that recognizes <think>...</think> blocks."""

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        end_tag = "</think>"
        start_tag = "<think>"

        if end_tag in current_text and end_tag not in previous_text:
            idx = current_text.index(end_tag) + len(end_tag)
            content_after = current_text[idx:]
            current_text[len(start_tag) : current_text.index(end_tag)]

            @dataclass
            class _Delta:
                reasoning_content: str | None = None
                content: str | None = None

            return _Delta(reasoning_content=None, content=content_after)

        if start_tag in current_text and end_tag not in current_text:
            reasoning_text = current_text[len(start_tag) :]
            prev_reasoning = previous_text[len(start_tag) :] if start_tag in previous_text else ""
            delta_r = reasoning_text[len(prev_reasoning) :]

            @dataclass
            class _Delta:
                reasoning_content: str | None = None
                content: str | None = None

            return _Delta(reasoning_content=delta_r, content=None)

        return None

    def extract_reasoning(self, text):
        end_tag = "</think>"
        start_tag = "<think>"
        if start_tag in text and end_tag in text:
            start = text.index(start_tag) + len(start_tag)
            end = text.index(end_tag)
            reasoning = text[start:end]
            content = text[end + len(end_tag) :]
            return reasoning, content
        if start_tag in text:
            return text[len(start_tag) :], None
        return None, text


@dataclass
class _FakeToolExtraction:
    tools_called: bool = False
    tool_calls: list | None = None
    content: str = ""


@dataclass
class _FakeDeltaMessage:
    content: str | None = None
    tool_calls: list | None = None


class _FakeToolParser:
    """Simulates a tool parser that detects <tool_call> blocks."""

    def __init__(self):
        self._streaming_calls = []

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request=None,
    ):
        self._streaming_calls.append(
            {
                "prev": previous_text,
                "curr": current_text,
                "delta": delta_text,
            }
        )
        if "<tool_call>" in current_text:
            return _FakeDeltaMessage(content=None, tool_calls=[{"name": "test"}])
        return _FakeDeltaMessage(content=delta_text)

    def extract_tool_calls(self, text, request):
        if "<tool_call>" in text:
            return _FakeToolExtraction(
                tools_called=True,
                tool_calls=[{"name": "test_fn"}],
                content="",
            )
        return _FakeToolExtraction(content=text)

    def is_buffering_protocol(self, current_text, delta_text):
        return "<tool" in current_text and "<tool_call>" not in current_text


class TestParsePhaseInit:
    def test_no_reasoning_starts_in_content(self):
        p = DelegatingParser()
        assert p.phase == ParsePhase.CONTENT

    def test_with_reasoning_starts_in_reasoning(self):
        p = DelegatingParser(reasoning_parser=_FakeReasoningParser())
        assert p.phase == ParsePhase.REASONING

    def test_with_tool_parser_starts_in_content(self):
        p = DelegatingParser(tool_parser=_FakeToolParser())
        assert p.phase == ParsePhase.CONTENT


class TestParseResult:
    def test_to_dict(self):
        r = ParseResult(
            delta_reasoning="think",
            delta_content="hello",
            accumulated_reasoning="think",
            accumulated_content="hello",
        )
        d = r.to_dict()
        assert d["delta_reasoning"] == "think"
        assert d["delta_content"] == "hello"
        assert d["tool_calls"] is None
        assert d["delta_tool_calls"] is None

    def test_default_values(self):
        r = ParseResult()
        assert r.delta_reasoning is None
        assert r.delta_content is None
        assert r.accumulated_reasoning == ""
        assert r.accumulated_content == ""
        assert r.phase == ParsePhase.CONTENT


class TestProcessDeltaNoParser:
    def test_content_only(self):
        p = DelegatingParser()
        result = p.process_delta("Hello", "Hello", [1, 2], "", [])
        assert result.delta_content == "Hello"
        assert result.accumulated_content == "Hello"

    def test_incremental_content(self):
        p = DelegatingParser()
        p.process_delta("Hello", "Hello", [1], "", [])
        r2 = p.process_delta("Hello World", " World", [1, 2], "Hello", [1])
        assert r2.delta_content == " World"
        assert r2.accumulated_content == "Hello World"


class TestProcessDeltaWithToolParser:
    def test_no_tool_call(self):
        p = DelegatingParser(tool_parser=_FakeToolParser())
        result = p.process_delta("Hello", "Hello", [1], "", [])
        assert result.delta_content == "Hello"
        assert result.delta_tool_calls is None

    def test_tool_call_detected(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
        p = DelegatingParser(tool_parser=_FakeToolParser(), tool_request=req)
        result = p.process_delta(
            "<tool_call>test</tool_call>",
            "<tool_call>test</tool_call>",
            [1, 2, 3],
            "",
            [],
        )
        assert result.delta_tool_calls is not None
        assert p.phase == ParsePhase.TOOL_CALL

    def test_buffering_phase(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
        p = DelegatingParser(tool_parser=_FakeToolParser(), tool_request=req)
        result = p.process_delta("<tool", "<tool", [1], "", [])
        assert p.phase == ParsePhase.BUFFERING
        assert result.delta_content == ""

    def test_tool_parser_view_reset_after_reasoning(self):
        p = DelegatingParser(
            reasoning_parser=_FakeReasoningParser(),
            tool_parser=_FakeToolParser(),
        )
        p.process_delta("<think>think", "<think>think", [1, 2], "", [])
        assert p.phase == ParsePhase.REASONING

        p.process_delta(
            "<think>think</think>content",
            "</think>content",
            [1, 2, 3, 4],
            "<think>think",
            [1, 2],
        )
        assert p._tool_previous_text == "" or p._tool_previous_text == "content"


class TestProcessFinal:
    def test_no_parsers(self):
        p = DelegatingParser()
        result = p.process_final("Hello World", [1, 2])
        assert result.accumulated_content == "Hello World"
        assert result.phase == ParsePhase.CONTENT

    def test_with_reasoning(self):
        p = DelegatingParser(reasoning_parser=_FakeReasoningParser())
        result = p.process_final("<think>reasoning</think>content", [1, 2, 3])
        assert result.accumulated_reasoning == "reasoning"
        assert result.accumulated_content == "content"

    def test_with_tool_calls(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
        p = DelegatingParser(tool_parser=_FakeToolParser(), tool_request=req)
        result = p.process_final("<tool_call>test</tool_call>", [1, 2])
        assert result.tool_calls is not None

    def test_reasoning_only(self):
        p = DelegatingParser(reasoning_parser=_FakeReasoningParser())
        result = p.process_final("<think>all reasoning, no close", [1, 2])
        assert result.accumulated_content == ""

    def test_no_reasoning_in_text(self):
        p = DelegatingParser(reasoning_parser=_FakeReasoningParser())
        result = p.process_final("Just plain content", [1, 2])
        assert result.accumulated_content == "Just plain content"


class TestIsToolsEnabled:
    def test_no_request(self):
        p = DelegatingParser(tool_parser=_FakeToolParser())
        assert not p._is_tools_enabled()

    def test_no_tools_in_request(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
        )
        p = DelegatingParser(tool_parser=_FakeToolParser(), tool_request=req)
        assert not p._is_tools_enabled()

    def test_tool_choice_none(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="none",
        )
        p = DelegatingParser(tool_parser=_FakeToolParser(), tool_request=req)
        assert not p._is_tools_enabled()

    def test_tools_enabled(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",
        )
        p = DelegatingParser(tool_parser=_FakeToolParser(), tool_request=req)
        assert p._is_tools_enabled()


class TestGetToolRequest:
    def test_creates_dummy_when_none(self):
        p = DelegatingParser(tool_parser=_FakeToolParser())
        req = p._get_tool_request()
        assert req.model == "dummy"

    def test_returns_existing_request(self):
        req = ChatCompletionRequest(
            model="my-model",
            messages=[ChatMessage(role="user", content="hi")],
        )
        p = DelegatingParser(tool_parser=_FakeToolParser(), tool_request=req)
        assert p._get_tool_request() is req
