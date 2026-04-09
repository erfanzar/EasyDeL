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

"""Unit tests for the DelegatingParser and its ParsePhase state machine."""

from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from easydel.inference.parsing import DelegatingParser, ParsePhase, ParseResult


class _StubReasoningParser:
    """Minimal reasoning parser that splits on <think>...</think> tags."""

    def __init__(self, tokenizer=None):
        self.model_tokenizer = tokenizer
        self.assume_reasoning = False

    def configure_prompt_context(self, prompt_text="", prompt_token_ids=()):
        pass

    def extract_reasoning(self, model_output, request=None):
        if "<think>" not in model_output:
            return None, model_output
        _before, after = model_output.split("<think>", 1)
        if "</think>" not in after:
            return after, None
        reasoning, content = after.split("</think>", 1)
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text="",
        current_text="",
        delta_text="",
        previous_token_ids=(),
        current_token_ids=(),
        delta_token_ids=(),
        request=None,
    ):
        if not delta_text:
            return None
        has_start_in_prev = "<think>" in previous_text
        has_end_in_prev = "</think>" in previous_text

        if has_end_in_prev:
            return DeltaMessage(content=delta_text)

        if "<think>" in delta_text and "</think>" in delta_text:
            after = delta_text.split("<think>", 1)[1]
            reasoning, content = after.split("</think>", 1)
            return DeltaMessage(
                reasoning_content=reasoning or None,
                content=content or None,
            )

        if "<think>" in delta_text:
            after = delta_text.split("<think>", 1)[1]
            return DeltaMessage(reasoning_content=after) if after else None

        if has_start_in_prev and "</think>" in delta_text:
            parts = delta_text.split("</think>", 1)
            return DeltaMessage(
                reasoning_content=parts[0] or None,
                content=parts[1] or None,
            )

        if has_start_in_prev:
            return DeltaMessage(reasoning_content=delta_text)

        return DeltaMessage(content=delta_text)


class _StubToolParser:
    """Minimal tool parser that detects <tool_call>...</tool_call> tags."""

    def __init__(self, tokenizer=None):
        self.model_tokenizer = tokenizer
        self.prev_tool_call_arr = []

    def extract_tool_calls(self, content, request):
        if "<tool_call>" in content:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[
                    ToolCall(
                        type="function",
                        function=FunctionCall(name="test_func", arguments='{"a": 1}'),
                    )
                ],
                content=None,
            )
        return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content)

    def extract_tool_calls_streaming(
        self,
        previous_text="",
        current_text="",
        delta_text="",
        previous_token_ids=(),
        current_token_ids=(),
        delta_token_ids=(),
        request=None,
    ):
        if "<tool_call>" in delta_text or "<tool_call>" in current_text:
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=DeltaFunctionCall(name="test_func", arguments='{"a":'),
                    )
                ]
            )
        return DeltaMessage(content=delta_text)

    def is_buffering_protocol(self, *, current_text="", delta_text=""):
        return "<tool_" in delta_text and "<tool_call>" not in delta_text


def _make_tool_request():
    return ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_func",
                        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
                    },
                }
            ],
        }
    )


def test_reasoning_only_streaming():
    dp = DelegatingParser(reasoning_parser=_StubReasoningParser())
    assert dp.phase == ParsePhase.REASONING

    prev_text = ""
    r = dp.process_delta("<think>I think", "<think>I think", "<think>I think", [], [])
    assert r.delta_reasoning == "I think"
    assert r.phase == ParsePhase.REASONING
    prev_text = "<think>I think"

    r = dp.process_delta("<think>I think more", " more", " more", prev_text, [])
    assert r.delta_reasoning == " more"
    prev_text = "<think>I think more"

    r = dp.process_delta("<think>I think more</think>answer", "</think>answer", "</think>answer", prev_text, [])
    assert dp.phase == ParsePhase.CONTENT
    assert r.accumulated_content is not None


def test_reasoning_only_final():
    dp = DelegatingParser(reasoning_parser=_StubReasoningParser())
    r = dp.process_final("<think>I think</think>the answer", [])
    assert r.accumulated_reasoning == "I think"
    assert r.accumulated_content == "the answer"


def test_no_reasoning_passes_through():
    dp = DelegatingParser(reasoning_parser=_StubReasoningParser())
    r = dp.process_final("just plain text", [])
    assert r.accumulated_reasoning == ""
    assert r.accumulated_content == "just plain text"


def test_tool_only_batch():
    dp = DelegatingParser(
        tool_parser=_StubToolParser(),
        tool_request=_make_tool_request(),
    )
    assert dp.phase == ParsePhase.CONTENT

    r = dp.process_final("<tool_call>test</tool_call>", [])
    assert r.tool_calls is not None
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].function.name == "test_func"


def test_tool_only_streaming():
    dp = DelegatingParser(
        tool_parser=_StubToolParser(),
        tool_request=_make_tool_request(),
    )
    r = dp.process_delta("<tool_call>", "<tool_call>", "<tool_call>", [], [])
    assert r.delta_tool_calls is not None
    assert dp.phase == ParsePhase.TOOL_CALL


def test_no_tool_text_passthrough():
    dp = DelegatingParser(
        tool_parser=_StubToolParser(),
        tool_request=_make_tool_request(),
    )
    r = dp.process_delta("hello world", "hello world", "hello world", [], [])
    assert r.delta_content == "hello world"
    assert r.delta_tool_calls is None


def test_combined_reasoning_then_tool():
    dp = DelegatingParser(
        reasoning_parser=_StubReasoningParser(),
        tool_parser=_StubToolParser(),
        tool_request=_make_tool_request(),
    )
    assert dp.phase == ParsePhase.REASONING

    r = dp.process_delta("<think>hmm", "<think>hmm", "<think>hmm", [], [])
    assert r.delta_reasoning == "hmm"
    assert dp.phase == ParsePhase.REASONING

    r = dp.process_final("<think>hmm</think><tool_call>test</tool_call>", [])
    assert r.accumulated_reasoning == "hmm"
    assert r.tool_calls is not None
    assert r.tool_calls[0].function.name == "test_func"


def test_combined_no_tools_in_content():
    dp = DelegatingParser(
        reasoning_parser=_StubReasoningParser(),
        tool_parser=_StubToolParser(),
        tool_request=_make_tool_request(),
    )
    r = dp.process_final("<think>hmm</think>just text", [])
    assert r.accumulated_reasoning == "hmm"
    assert r.accumulated_content == "just text"
    assert r.tool_calls is None


def test_buffering_suppresses_content():
    dp = DelegatingParser(
        tool_parser=_StubToolParser(),
        tool_request=_make_tool_request(),
    )
    r = dp.process_delta("<tool_", "<tool_", "<tool_", [], [])
    assert r.delta_content == ""
    assert dp.phase == ParsePhase.BUFFERING


def test_no_parsers_passthrough():
    dp = DelegatingParser()
    r = dp.process_delta("hello", "hello", "hello", [], [])
    assert r.accumulated_content == "hello"
    assert r.delta_content == "hello"
    assert r.delta_reasoning is None

    r = dp.process_final("hello world", [])
    assert r.accumulated_content == "hello world"


def test_parse_result_to_dict():
    r = ParseResult(
        delta_reasoning="think",
        delta_content="text",
        accumulated_reasoning="think",
        accumulated_content="text",
        tool_calls=[ToolCall(type="function", function=FunctionCall(name="f", arguments="{}"))],
        delta_tool_calls=None,
        phase=ParsePhase.CONTENT,
    )
    d = r.to_dict()
    assert d["delta_reasoning"] == "think"
    assert d["delta_content"] == "text"
    assert d["tool_calls"] is not None
    assert "phase" not in d  # phase is not in the dict


class _FailingReasoningParser(_StubReasoningParser):
    def extract_reasoning_streaming(self, **kwargs):
        raise RuntimeError("boom")

    def extract_reasoning(self, model_output, request=None):
        raise RuntimeError("boom")


def test_reasoning_error_falls_through_to_content():
    dp = DelegatingParser(reasoning_parser=_FailingReasoningParser())
    r = dp.process_delta("hello", "hello", "hello", [], [])
    assert r.accumulated_content == "hello"
    assert dp.phase == ParsePhase.CONTENT


def test_reasoning_error_final_falls_through():
    dp = DelegatingParser(reasoning_parser=_FailingReasoningParser())
    r = dp.process_final("hello", [])
    assert r.accumulated_content == "hello"


def test_tool_previous_text_resets_at_reasoning_boundary():
    dp = DelegatingParser(
        reasoning_parser=_StubReasoningParser(),
        tool_parser=_StubToolParser(),
        tool_request=_make_tool_request(),
    )

    dp.process_delta("<think>r", "<think>r", "<think>r", [], [])
    assert dp.phase == ParsePhase.REASONING
    assert dp._tool_previous_text == ""

    dp.process_delta("<think>r</think>content", "</think>content", "</think>content", [], [])
    assert dp._tool_previous_text != "<think>r</think>content"


class _StripReasoningParser(_StubReasoningParser):
    """Parser that strips whitespace like the real BaseThinkingReasoningParser."""

    def extract_reasoning(self, model_output, request=None):
        if "<think>" not in model_output:
            return None, model_output
        _before, after = model_output.split("<think>", 1)
        if "</think>" not in after:
            return after.strip(), None
        reasoning, content = after.split("</think>", 1)
        return reasoning.strip() or None, content.strip() or None


def test_strip_normalization_reassembled_deltas_match_visible_snapshot():
    dp = DelegatingParser(reasoning_parser=_StripReasoningParser())
    pieces = [
        "<think>",
        "short",
        "</think>\n\nHello!",
        " How",
        " are",
        " you?",
    ]

    accumulated_text = ""
    previous_text = ""
    token_ids: list[int] = []
    previous_token_ids: list[int] = []
    emitted_text = ""

    for index, piece in enumerate(pieces, start=1):
        accumulated_text += piece
        token_ids.append(index)
        result = dp.process_delta(
            accumulated_text,
            piece,
            list(token_ids),
            previous_text,
            list(previous_token_ids),
        )

        if result.delta_content:
            emitted_text += result.delta_content

        assert emitted_text == result.accumulated_content
        previous_text = accumulated_text
        previous_token_ids = list(token_ids)

    assert emitted_text == "Hello! How are you?"


def test_strip_normalization_does_not_cause_content_mismatch():
    """When batch extraction strips whitespace but streaming didn't,
    the accumulated_content should stay aligned with previous content."""
    dp = DelegatingParser(reasoning_parser=_StripReasoningParser())

    r1 = dp.process_delta(
        "<think>r</think>hello ",
        "<think>r</think>hello ",
        "<think>r</think>hello ",
        "",
        [],
    )
    assert dp.phase == ParsePhase.CONTENT
    prev_content = r1.accumulated_content

    r2 = dp.process_delta(
        "<think>r</think>hello x",
        "x",
        "x",
        "<think>r</think>hello ",
        [],
    )
    assert len(r2.accumulated_content) >= len(prev_content)


def test_equal_length_content_rewrite_keeps_previous():
    """When batch extraction returns same-length but different content,
    keep the previous accumulated content to avoid alignment issues."""
    dp = DelegatingParser(reasoning_parser=_StripReasoningParser())

    dp.phase = ParsePhase.CONTENT
    dp._accumulated_content = "hello_world"
    dp._accumulated_reasoning = ""

    dp.process_delta(
        "<think></think>hello world",
        " world",
        " world",
        "<think></think>hello_",
        [],
    )


def test_finish_reason_priority_tool_calls_beats_stop():
    """In _resolve_public_finish_reason, tool_calls should beat stop."""
    from easydel.inference.esurge.mixins.parsing import EngineParsingMixin

    class _FakeOutput:
        def __init__(self, fr):
            self.finish_reason = fr

    assert (
        EngineParsingMixin._resolve_public_finish_reason([_FakeOutput("stop"), _FakeOutput("tool_calls")])
        == "tool_calls"
    )

    assert (
        EngineParsingMixin._resolve_public_finish_reason([_FakeOutput("tool_calls"), _FakeOutput("stop")])
        == "tool_calls"
    )

    assert (
        EngineParsingMixin._resolve_public_finish_reason([_FakeOutput("length"), _FakeOutput("tool_calls")]) == "length"
    )

    assert EngineParsingMixin._resolve_public_finish_reason([_FakeOutput("abort")]) == "abort"

    assert EngineParsingMixin._resolve_public_finish_reason([_FakeOutput("stop")]) == "stop"


def test_tool_parser_without_request_passes_content():
    """Tool parser with no tool_request should pass content through."""
    dp = DelegatingParser(
        tool_parser=_StubToolParser(),
        tool_request=None,  # No tools configured
    )
    r = dp.process_delta("hello", "hello", "hello", [], [])
    assert r.accumulated_content == "hello"


def test_tool_parser_with_tool_choice_none():
    """Tool parser with tool_choice='none' should pass content through."""
    from easydel.inference.openai_api_modules import ChatCompletionRequest

    req = ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "none",
        }
    )
    dp = DelegatingParser(
        tool_parser=_StubToolParser(),
        tool_request=req,
    )
    r = dp.process_delta("hello", "hello", "hello", [], [])
    assert r.accumulated_content == "hello"
    assert dp.phase == ParsePhase.CONTENT
