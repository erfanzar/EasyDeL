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

"""Parser-certified content is independent of protocol buffering and tool deltas."""

import pytest
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from easydel.inference.parsing import DelegatingParser, ParsePhase
from easydel.inference.tools.abstract_tool import ToolParser
from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

pytestmark = pytest.mark.cpu_ok


class _ScriptedParser(ToolParser):
    """Model a parser that certifies safe text while withholding protocol text."""

    def __init__(self, steps, final):
        super().__init__(object())
        self.steps = iter(steps)
        self.final = final
        self.buffering = False

    def extract_tool_calls_streaming(self, **kwargs):
        message, self.buffering = next(self.steps)
        return message

    def is_buffering_protocol(self, **kwargs):
        return self.buffering

    def extract_tool_calls(self, model_output, request):
        return self.final


def _delegating(steps, final=None):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        }
    )
    return DelegatingParser(
        tool_parser=_ScriptedParser(
            steps, final or ExtractedToolCallInformation(tools_called=False, tool_calls=[], content="")
        ),
        tool_request=request,
    )


def _call_delta():
    return DeltaToolCall(index=0, id="stream_id", type="function", function=DeltaFunctionCall(name="f", arguments="{}"))


@pytest.mark.parametrize("buffering", [False, True])
def test_tool_delta_and_safe_content_both_survive(buffering):
    parser = _delegating([(DeltaMessage(content="Before. ", tool_calls=[_call_delta()]), buffering)])
    result = parser.process_delta("Before. <tool_call>f{}</tool_call>", "raw protocol", [], "", [])
    assert result.delta_content == result.accumulated_content == "Before. "
    assert result.delta_tool_calls[0].function.name == "f"
    assert parser.streamed_tool_call_id(0) == "stream_id"
    assert result.phase == ParsePhase.TOOL_CALL


def test_safe_preamble_survives_pending_marker_and_disambiguation_releases_buffer():
    parser = _delegating(
        [
            (DeltaMessage(content="ordinary "), True),
            (DeltaMessage(content="<tool_box"), False),
            (DeltaMessage(content="!"), False),
        ]
    )
    first = parser.process_delta("ordinary <tool_", "ordinary <tool_", [], "", [])
    assert first.delta_content == first.accumulated_content == "ordinary "
    assert first.phase == ParsePhase.BUFFERING
    second = parser.process_delta("ordinary <tool_box", "box", [], "ordinary <tool_", [])
    assert second.delta_content == "<tool_box"
    assert second.accumulated_content == "ordinary <tool_box"
    assert second.phase == ParsePhase.CONTENT
    third = parser.process_delta("ordinary <tool_box!", "!", [], "ordinary <tool_box", [])
    assert third.delta_content == "!"
    assert third.accumulated_content == "ordinary <tool_box!"


def test_empty_buffer_can_resolve_to_ordinary_content():
    parser = _delegating([(None, True), (DeltaMessage(content="<tool_box"), False)])
    pending = parser.process_delta("<tool_", "<tool_", [], "", [])
    assert not pending.delta_content
    assert pending.accumulated_content == ""
    assert pending.phase == ParsePhase.BUFFERING
    resolved = parser.process_delta("<tool_box", "box", [], "<tool_", [])
    assert resolved.delta_content == resolved.accumulated_content == "<tool_box"
    assert resolved.phase == ParsePhase.CONTENT


def test_explicit_content_after_tool_call_is_not_suppressed_by_old_phase():
    parser = _delegating([(DeltaMessage(tool_calls=[_call_delta()]), True), (DeltaMessage(content="After."), False)])
    first = parser.process_delta("<tool_call>f{}", "<tool_call>f{}", [], "", [])
    assert first.phase == ParsePhase.TOOL_CALL
    assert not first.delta_content
    result = parser.process_delta("<tool_call>f{}</tool_call>After.", "</tool_call>After.", [], "<tool_call>f{}", [])
    assert result.delta_content == result.accumulated_content == "After."
    assert result.phase == ParsePhase.CONTENT


@pytest.mark.parametrize("streamed", ["", "Before. "])
def test_final_tool_calls_preserve_safe_content_and_only_emit_missing_suffix(streamed):
    final = ExtractedToolCallInformation(
        tools_called=True,
        tool_calls=[ToolCall(id="batch_id", function=FunctionCall(name="f", arguments="{}"))],
        content="Before. After.",
    )
    parser = _delegating([(DeltaMessage(content=streamed, tool_calls=[_call_delta()]), True)], final)
    first = parser.process_delta("raw protocol", "raw protocol", [], "", [])
    result = parser.process_final("raw protocol", [])
    assert (first.delta_content or "") + (result.delta_content or "") == "Before. After."
    assert result.accumulated_content == "Before. After."
    assert result.tool_calls[0].id == "stream_id"
    assert result.delta_tool_calls is None
    assert not parser.process_final("raw protocol", []).delta_content


@pytest.mark.parametrize("batch_content", ["Before. ", ""])
def test_final_success_does_not_retract_certified_streamed_suffix(batch_content):
    final = ExtractedToolCallInformation(
        tools_called=True,
        tool_calls=[ToolCall(function=FunctionCall(name="f", arguments="{}"))],
        content=batch_content,
    )
    parser = _delegating([(DeltaMessage(content="Before. After.", tool_calls=[_call_delta()]), True)], final)
    parser.process_delta("raw protocol", "raw protocol", [], "", [])
    result = parser.process_final("raw protocol", [])
    assert result.accumulated_content == "Before. After."
    assert not result.delta_content
    assert result.tool_calls[0].id == "stream_id"


@pytest.mark.parametrize("trailing_content", [False, True])
def test_failed_final_batch_does_not_override_recognized_tool_boundary(trailing_content):
    raw = "Before. <tool_call>unfinished"
    steps = [(DeltaMessage(content="Before. ", tool_calls=[_call_delta()]), True)]
    if trailing_content:
        steps.append((DeltaMessage(content="After."), False))
    parser = _delegating(steps, ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=raw))
    parser.process_delta(raw, raw, [], "", [])
    if trailing_content:
        parser.process_delta(raw, "", [], raw, [])
    result = parser.process_final(raw, [])
    assert result.accumulated_content == "Before. " + ("After." if trailing_content else "")
    assert not result.delta_content


def test_raising_final_batch_preserves_recognized_tool_boundary(monkeypatch):
    parser = _delegating([(DeltaMessage(content="Before. ", tool_calls=[_call_delta()]), True)])
    raw = "Before. <tool_call>unfinished"
    parser.process_delta(raw, raw, [], "", [])

    def fail_batch(model_output, request):
        raise ValueError("unfinished protocol")

    monkeypatch.setattr(parser.tool_parser, "extract_tool_calls", fail_batch)
    result = parser.process_final(raw, [])
    assert result.accumulated_content == "Before. "
    assert not result.delta_content


class _CharacterTokenizer:
    """Deterministic text-only tokenizer for real parser integration."""

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.mark.parametrize(
    "chunks, expected, tool_called",
    [
        (["ordinary <tool_", "box", "!"], "ordinary <tool_box!", False),
        (["<tool_", "box", "!"], "<tool_box!", False),
        (['Before. <tool_call>{"name":"f","arguments":{}}</tool_call>'], "Before. ", True),
        (["Before. ", "<tool_", 'call>{"name":"f","arguments":{}}</tool_call>'], "Before. ", True),
        (['Before. <tool_call>{"name":"f","arguments":{}}</tool_call>After.'], "Before. After.", True),
        (["Before. ", '<tool_call>{"name":"f","arguments":{}}</tool_call>', "After."], "Before. After.", True),
    ],
)
def test_real_hermes_preserves_preambles_and_releases_false_prefixes_before_final(chunks, expected, tool_called):
    parser = _delegating([])
    parser = DelegatingParser(tool_parser=HermesToolParser(_CharacterTokenizer()), tool_request=parser.tool_request)
    previous = ""
    visible = ""
    tool_deltas = []
    for chunk in chunks:
        current = previous + chunk
        result = parser.process_delta(current, chunk, [], previous, [])
        visible += result.delta_content or ""
        tool_deltas.extend(result.delta_tool_calls or [])
        previous = current
    assert visible == expected
    assert result.accumulated_content == expected
    if not tool_called:
        assert result.phase == ParsePhase.CONTENT
    final = parser.process_final(previous, [])
    assert not final.delta_content
    assert final.accumulated_content == expected
    assert bool(final.tool_calls) is tool_called
    if tool_called:
        assert [call.function.name for call in tool_deltas if call.function.name] == ["f"]


@pytest.mark.parametrize("split", [False, True])
def test_real_hermes_failed_batch_keeps_partial_protocol_out_of_content(split):
    parser = _delegating([])
    parser = DelegatingParser(tool_parser=HermesToolParser(_CharacterTokenizer()), tool_request=parser.tool_request)
    raw = 'Before. <tool_call>{"name":"f","arguments":{"x":'
    chunks = ["Before. ", raw[len("Before. ") :]] if split else [raw]
    previous = ""
    visible = ""
    tool_deltas = []
    for chunk in chunks:
        current = previous + chunk
        result = parser.process_delta(current, chunk, [], previous, [])
        visible += result.delta_content or ""
        tool_deltas.extend(result.delta_tool_calls or [])
        previous = current
    assert visible == result.accumulated_content == "Before. "
    assert any(call.function.name == "f" for call in tool_deltas)
    final = parser.process_final(raw, [])
    assert not final.delta_content
    assert final.accumulated_content == visible
    assert not parser.process_final(raw, []).delta_content
